
import numpy as np
import pandas as pd
from scipy.optimize import minimize

try:
    from IPython.display import display
except Exception:
    display = print

INPUT_FILE = "preqin_company_sequences_long.csv"

OUT_COUNTS = "hmm_expected_transition_counts.csv"
OUT_PROBS  = "hmm_expected_transition_probabilities.csv"
OUT_EMISS  = "hmm_emission_probabilities.csv"
OUT_PARAMS = "hmm_transition_logit_params.csv"
OUT_PI     = "hmm_initial_state_distribution.csv"
OUT_ABSORB = "hmm_absorption_probabilities.csv"
OUT_SIZE_TABLE = "hmm_transition_probs_at_sizes.csv"

RANDOM_SEED = 7

STATES = ["Seed", "Series A", "Series B", "Series C", "Series D", "Series E", "Exit", "Fail"]
STATE_INDEX = {s: i for i, s in enumerate(STATES)}

PROGRESSION_STATES = ["Seed", "Series A", "Series B", "Series C", "Series D", "Series E"]

MAX_ITER = 40
TOL = 1e-4
EMISSION_DIRICHLET = 0.25
RIDGE_L2 = 0.1
VERBOSE = True

# Deal-size checkpoints (USD MN)
SIZE_POINTS_USD_MN = [5, 10, 25, 50, 100]


def logsumexp(a: np.ndarray, axis=None) -> np.ndarray:
    a_max = np.max(a, axis=axis, keepdims=True)
    out = a_max + np.log(np.sum(np.exp(a - a_max), axis=axis, keepdims=True))
    if axis is not None:
        out = np.squeeze(out, axis=axis)
    return out


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - np.max(z)
    e = np.exp(z)
    return e / np.sum(e)


def build_allowed_next_states() -> dict[int, list[int]]:
    allowed = {}
    for s in PROGRESSION_STATES:
        i = STATE_INDEX[s]
        nxt = min(i + 1, STATE_INDEX["Series E"])
        allowed[i] = sorted({i, nxt, STATE_INDEX["Exit"], STATE_INDEX["Fail"]})
    allowed[STATE_INDEX["Exit"]] = [STATE_INDEX["Exit"]]
    allowed[STATE_INDEX["Fail"]] = [STATE_INDEX["Fail"]]
    return allowed


ALLOWED_NEXT = build_allowed_next_states()


def standardize_feature(df: pd.DataFrame) -> tuple[np.ndarray, float, float]:
    r = df["log_raise"].to_numpy()
    mask = np.isfinite(r)
    if mask.sum() == 0:
        mu, sd = 0.0, 1.0
    else:
        mu = float(np.mean(r[mask]))
        sd = float(np.std(r[mask]))
        if sd < 1e-8:
            sd = 1.0
    r_std = np.zeros_like(r, dtype=float)
    r_std[mask] = (r[mask] - mu) / sd
    r_std[~mask] = 0.0
    return r_std, mu, sd


def initialize_emissions(obs_vocab: list[str]) -> np.ndarray:
    np.random.seed(RANDOM_SEED)
    n_states = len(STATES)
    n_obs = len(obs_vocab)
    obs_index = {o: k for k, o in enumerate(obs_vocab)}

    B = np.full((n_states, n_obs), 1.0 / n_obs, dtype=float)

    for stage in ["Seed", "Series A", "Series B", "Series C", "Series D", "Series E"]:
        if stage in obs_index:
            si = STATE_INDEX[stage]
            ok = obs_index[stage]
            B[si, :] *= 0.2
            B[si, ok] += 0.8

    if "EXITLIKE" in obs_index:
        ok = obs_index["EXITLIKE"]
        B[STATE_INDEX["Exit"], :] *= 0.05
        B[STATE_INDEX["Exit"], ok] += 0.95

    B = B / B.sum(axis=1, keepdims=True)
    return B


def initialize_pi() -> np.ndarray:
    pi = np.zeros(len(STATES), dtype=float)
    pi[STATE_INDEX["Seed"]] = 0.50
    pi[STATE_INDEX["Series A"]] = 0.35
    pi[STATE_INDEX["Series B"]] = 0.10
    pi[STATE_INDEX["Series C"]] = 0.03
    pi[STATE_INDEX["Series D"]] = 0.01
    pi[STATE_INDEX["Series E"]] = 0.01
    pi = pi / pi.sum()
    return pi


def transition_probs_for_time(i: int, x: np.ndarray, params_i: np.ndarray, choices: list[int]) -> np.ndarray:
    m = len(choices)
    if m == 1:
        return np.array([1.0], dtype=float)

    logits = np.zeros(m, dtype=float)  # reference class = choices[0]
    for k in range(1, m):
        logits[k] = float(np.dot(params_i[k - 1], x))
    return softmax(logits)


def build_transition_matrices_for_sequence(r_seq_std: np.ndarray,
                                          miss_seq: np.ndarray,
                                          params: dict[int, np.ndarray]) -> list[np.ndarray]:
    n_states = len(STATES)
    T = len(r_seq_std)
    A_list = []
    for t in range(1, T):
        x = np.array([1.0, float(r_seq_std[t]), float(miss_seq[t])], dtype=float)
        A = np.zeros((n_states, n_states), dtype=float)
        for i in range(n_states):
            choices = ALLOWED_NEXT[i]
            if len(choices) == 1:
                A[i, choices[0]] = 1.0
                continue
            probs = transition_probs_for_time(i, x, params[i], choices)
            for idx, j in enumerate(choices):
                A[i, j] = probs[idx]
        A = A / A.sum(axis=1, keepdims=True)
        A_list.append(A)
    return A_list


def fit_weighted_multinomial_logit(X: np.ndarray, W: np.ndarray, ridge_l2: float) -> np.ndarray:
    n, p = X.shape
    m = W.shape[1]
    if m == 1:
        return np.zeros((0, p), dtype=float)

    def unpack(theta_flat):
        return theta_flat.reshape((m - 1, p))

    def nll(theta_flat):
        Theta = unpack(theta_flat)
        logits = np.zeros((n, m), dtype=float)
        for k in range(1, m):
            logits[:, k] = X @ Theta[k - 1].T

        logits_max = np.max(logits, axis=1, keepdims=True)
        lse = logits_max + np.log(np.sum(np.exp(logits - logits_max), axis=1, keepdims=True))
        logp = logits - lse
        loss = -np.sum(W * logp) + 0.5 * ridge_l2 * np.sum(Theta * Theta)
        return float(loss)

    def grad(theta_flat):
        Theta = unpack(theta_flat)
        logits = np.zeros((n, m), dtype=float)
        for k in range(1, m):
            logits[:, k] = X @ Theta[k - 1].T

        logits_max = np.max(logits, axis=1, keepdims=True)
        expz = np.exp(logits - logits_max)
        P = expz / np.sum(expz, axis=1, keepdims=True)

        row_w = np.sum(W, axis=1, keepdims=True)
        G = np.zeros((m - 1, p), dtype=float)
        for k in range(1, m):
            diff = (row_w[:, 0] * P[:, k] - W[:, k])
            G[k - 1, :] = X.T @ diff
        G += ridge_l2 * Theta
        return G.reshape(-1)

    theta0 = np.zeros((m - 1) * p, dtype=float)
    res = minimize(nll, theta0, jac=grad, method="L-BFGS-B")
    return unpack(res.x)


def compute_absorption_probabilities(P: np.ndarray, exit_idx: int, fail_idx: int) -> pd.DataFrame:
    n = P.shape[0]
    absorbing = [exit_idx, fail_idx]
    transient = [i for i in range(n) if i not in absorbing]

    Q = P[np.ix_(transient, transient)]
    R = P[np.ix_(transient, absorbing)]
    I = np.eye(len(transient))
    try:
        N = np.linalg.inv(I - Q)
    except np.linalg.LinAlgError:
        N = np.linalg.pinv(I - Q)
    B = N @ R

    prob_exit = np.zeros(n)
    prob_fail = np.zeros(n)
    for idx, s in enumerate(transient):
        prob_exit[s] = B[idx, 0]
        prob_fail[s] = B[idx, 1]

    prob_exit[exit_idx] = 1.0
    prob_fail[exit_idx] = 0.0
    prob_exit[fail_idx] = 0.0
    prob_fail[fail_idx] = 1.0

    return pd.DataFrame({"state": STATES, "prob_absorb_exit": prob_exit, "prob_absorb_fail": prob_fail})


def transition_probs_at_size_table(params: dict[int, np.ndarray], r_mu: float, r_sd: float) -> pd.DataFrame:
    """
    Create a table of P(next_state | current_state, size) for selected sizes.
    Uses miss=0.
    """
    rows = []
    for size in SIZE_POINTS_USD_MN:
        r = np.log1p(size)  # log_raise in USD MN space (matching preprocessing)
        r_std = 0.0 if r_sd < 1e-8 else (r - r_mu) / r_sd
        x = np.array([1.0, float(r_std), 0.0], dtype=float)

        for from_state in PROGRESSION_STATES:
            i = STATE_INDEX[from_state]
            choices = ALLOWED_NEXT[i]
            probs = transition_probs_for_time(i, x, params[i], choices)
            for idx, j in enumerate(choices):
                rows.append({
                    "from_state": from_state,
                    "deal_size_usd_mn": size,
                    "to_state": STATES[j],
                    "prob": float(probs[idx]),
                })

    return pd.DataFrame(rows)


def main():
    np.random.seed(RANDOM_SEED)

    df = pd.read_csv(INPUT_FILE)
    required = ["company_id", "t", "obs_token", "log_raise"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Input missing required columns: {missing}")

    df = df.sort_values(["company_id", "t"], kind="mergesort").reset_index(drop=True)

    obs_vocab = sorted(df["obs_token"].astype(str).unique().tolist())
    obs_index = {o: k for k, o in enumerate(obs_vocab)}
    df["obs_idx"] = df["obs_token"].map(obs_index).astype(int)

    r_std, r_mu, r_sd = standardize_feature(df)
    miss = (~np.isfinite(df["log_raise"].to_numpy())).astype(float)
    df["r_std"] = r_std
    df["r_miss"] = miss

    sequences = []
    for cid, g in df.groupby("company_id", sort=False):
        sequences.append((
            int(cid),
            g["obs_idx"].to_numpy(dtype=int),
            g["r_std"].to_numpy(dtype=float),
            g["r_miss"].to_numpy(dtype=float),
        ))

    n_states = len(STATES)
    n_obs = len(obs_vocab)

    B = initialize_emissions(obs_vocab)
    pi = initialize_pi()

    p = 3
    params = {}
    for i in range(n_states):
        m_i = len(ALLOWED_NEXT[i])
        params[i] = np.zeros((max(m_i - 1, 0), p), dtype=float)

    prev_ll = None
    for it in range(1, MAX_ITER + 1):
        pi_acc = np.zeros(n_states, dtype=float)
        emiss_counts = np.zeros((n_states, n_obs), dtype=float)
        xi_total = np.zeros((n_states, n_states), dtype=float)
        ll_total = 0.0

        trans_X = {i: [] for i in range(n_states)}
        trans_W = {i: [] for i in range(n_states)}

        for _, obs_seq, r_seq, m_seq in sequences:
            T = len(obs_seq)
            if T == 0:
                continue

            A_list = build_transition_matrices_for_sequence(r_seq, m_seq, params)
            logB = np.log(np.maximum(B[:, obs_seq], 1e-300))
            log_pi = np.log(np.maximum(pi, 1e-300))

            log_alpha = np.full((T, n_states), -np.inf)
            log_alpha[0, :] = log_pi + logB[:, 0]

            for t in range(1, T):
                logA = np.log(np.maximum(A_list[t - 1], 1e-300))
                tmp = log_alpha[t - 1, :][:, None] + logA
                log_alpha[t, :] = logB[:, t] + logsumexp(tmp, axis=0)

            loglik = float(logsumexp(log_alpha[T - 1, :], axis=0))
            ll_total += loglik

            log_beta = np.full((T, n_states), -np.inf)
            log_beta[T - 1, :] = 0.0
            for t in range(T - 2, -1, -1):
                logA = np.log(np.maximum(A_list[t], 1e-300))
                tmp = logA + (logB[:, t + 1] + log_beta[t + 1, :])[None, :]
                log_beta[t, :] = logsumexp(tmp, axis=1)

            log_gamma = log_alpha + log_beta - loglik
            gamma = np.exp(log_gamma)

            pi_acc += gamma[0, :]
            for t in range(T):
                emiss_counts[:, obs_seq[t]] += gamma[t, :]

            for t in range(1, T):
                logA = np.log(np.maximum(A_list[t - 1], 1e-300))
                log_xi = (
                    log_alpha[t - 1, :][:, None]
                    + logA
                    + logB[:, t][None, :]
                    + log_beta[t, :][None, :]
                    - loglik
                )
                xi = np.exp(log_xi)
                xi_total += xi

                x_t = np.array([1.0, float(r_seq[t]), float(m_seq[t])], dtype=float)
                for i in range(n_states):
                    choices = ALLOWED_NEXT[i]
                    if len(choices) <= 1:
                        continue
                    w = np.array([xi[i, j] for j in choices], dtype=float)
                    if w.sum() > 1e-12:
                        trans_X[i].append(x_t)
                        trans_W[i].append(w)

        # M-step
        pi = np.maximum(pi_acc, 1e-300)
        pi = pi / pi.sum()

        B = emiss_counts + EMISSION_DIRICHLET
        B = B / B.sum(axis=1, keepdims=True)

        for i in range(n_states):
            if len(ALLOWED_NEXT[i]) <= 1:
                continue
            X_i = np.array(trans_X[i], dtype=float)
            W_i = np.array(trans_W[i], dtype=float)
            if X_i.shape[0] == 0:
                continue
            params[i] = fit_weighted_multinomial_logit(X_i, W_i, ridge_l2=RIDGE_L2)

        if VERBOSE:
            print(f"EM iter {it:02d} | total log-likelihood: {ll_total:,.3f}")

        if prev_ll is not None and abs(ll_total - prev_ll) < TOL * (1.0 + abs(prev_ll)):
            if VERBOSE:
                print("Converged.")
            break
        prev_ll = ll_total

    # Expected transition matrices
    counts = xi_total
    probs = counts / np.maximum(counts.sum(axis=1, keepdims=True), 1e-300)

    df_counts = pd.DataFrame(counts, index=STATES, columns=STATES)
    df_probs  = pd.DataFrame(probs,  index=STATES, columns=STATES)
    df_emiss  = pd.DataFrame(B, index=STATES, columns=sorted(obs_vocab))
    df_pi     = pd.DataFrame({"state": STATES, "pi": pi})
    df_abs    = compute_absorption_probabilities(probs, STATE_INDEX["Exit"], STATE_INDEX["Fail"])

    # Size-conditioned transition table
    df_size = transition_probs_at_size_table(params, r_mu=r_mu, r_sd=r_sd)

    print("\n=== Expected Transition COUNT Matrix (posterior-weighted) ===")
    display(df_counts.style.format("{:,.2f}"))

    print("\n=== Expected Transition PROBABILITY Matrix (row-normalized) ===")
    display(df_probs.style.format("{:.4f}"))

    print("\n=== Absorption Probabilities (from marginal transition matrix) ===")
    display(df_abs.style.format({"prob_absorb_exit": "{:.4f}", "prob_absorb_fail": "{:.4f}"}))

    print("\n=== Transition probabilities at specific deal sizes (USD MN) ===")
    # Pivot into a compact view: rows = from_state + size, columns = to_state
    pivot = df_size.pivot_table(index=["from_state", "deal_size_usd_mn"], columns="to_state", values="prob", aggfunc="first").fillna(0.0)
    display(pivot.style.format("{:.4f}"))

    print("\n=== Emission Probabilities P(obs_token | state) ===")
    display(df_emiss.style.format("{:.4f}"))

    df_counts.to_csv(OUT_COUNTS)
    df_probs.to_csv(OUT_PROBS)
    df_emiss.to_csv(OUT_EMISS)
    df_pi.to_csv(OUT_PI, index=False)
    df_abs.to_csv(OUT_ABSORB, index=False)
    df_size.to_csv(OUT_SIZE_TABLE, index=False)

    # Save transition logit parameters
    feature_names = ["intercept", "r_std", "r_missing"]
    rows = []
    for i in range(len(STATES)):
        choices = ALLOWED_NEXT[i]
        if len(choices) <= 1:
            continue
        ref = choices[0]
        Theta = params[i]
        for k in range(1, len(choices)):
            j = choices[k]
            for f in range(3):
                rows.append({
                    "from_state": STATES[i],
                    "to_state": STATES[j],
                    "reference_to_state": STATES[ref],
                    "feature": feature_names[f],
                    "coef": float(Theta[k - 1, f]),
                })
    pd.DataFrame(rows).to_csv(OUT_PARAMS, index=False)

    print("\nSaved output CSVs:")
    for f in [OUT_COUNTS, OUT_PROBS, OUT_EMISS, OUT_PARAMS, OUT_PI, OUT_ABSORB, OUT_SIZE_TABLE]:
        print(" -", f)


if __name__ == "__main__":
    main()
