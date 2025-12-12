import pandas as pd
import numpy as np
import os

def load_company_sequences(path):
    
    # Load as CSV or Excel depending on extension
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        df = pd.read_csv(path)
    else:
        df = pd.read_excel(path)

    # Identify stage columns in order
    stage_cols = [c for c in df.columns if c.startswith("stage_")]
    # Sort by numeric suffix to ensure correct order
    stage_cols = sorted(stage_cols, key=lambda x: int(x.split("_")[1]))

    sequences = []
    company_ids = []

    for _, row in df.iterrows():
        stages = []
        for c in stage_cols:
            val = row[c]
            if pd.isna(val):
                continue
            stages.append(str(val))
        if len(stages) >= 1:
            sequences.append(stages)
            company_ids.append(row["PORTFOLIO COMPANY ID"])

    return sequences, company_ids


def estimate_transition_matrix(sequences, state_order=None, smoothing=0.0):
    
    # Infer unique states if not provided
    if state_order is None:
        uniq_states = set()
        for seq in sequences:
            for s in seq:
                uniq_states.add(s)
        states = sorted(uniq_states)
    else:
        states = list(state_order)

    n_states = len(states)
    state_to_idx = {s: i for i, s in enumerate(states)}

    # Count transitions
    counts = np.zeros((n_states, n_states), dtype=float)

    for seq in sequences:
        # consecutive pairs (s_t, s_{t+1})
        for t in range(len(seq) - 1):
            i = state_to_idx[seq[t]]
            j = state_to_idx[seq[t + 1]]
            counts[i, j] += 1.0

    # Apply smoothing and normalize rows
    counts_smoothed = counts + smoothing
    row_sums = counts_smoothed.sum(axis=1, keepdims=True)

    # To avoid division by zero (e.g. for states with no outgoing transitions),
    # we handle rows with zero total separately.
    P = np.zeros_like(counts_smoothed)
    for i in range(n_states):
        if row_sums[i, 0] > 0:
            P[i, :] = counts_smoothed[i, :] / row_sums[i, 0]
        else:
            # If a state has no observed outgoing transitions (e.g. pure absorbing),
            # we can set it to be self-absorbing with probability 1.
            P[i, i] = 1.0

    return states, counts, P


def main():
    # 1. Load sequences from the wide file
    sequences, company_ids = load_company_sequences(INPUT_FILE)
    print(f"Loaded {len(sequences)} company sequences from {INPUT_FILE}")

    # 2. Define the state order explicitly to match your model
    # If your canonical stages are exactly these:
    state_order = ["Seed", "Series A", "Series B", "Series C",
                   "Series D", "Series E", "Exit"]

    # 3. Estimate transition probabilities
    states, counts, P = estimate_transition_matrix(
        sequences,
        state_order=state_order,
        smoothing=0.0  # you could set e.g. 1.0 for Laplace smoothing if desired
    )

    # 4. Print transition counts and matrix nicely
    print("\nStates (in order):")
    print(states)

    print("\nTransition count matrix (rows = from, cols = to):")
    counts_df = pd.DataFrame(counts, index=states, columns=states)
    print(counts_df)

    print("\nEstimated transition probability matrix P(j | i):")
    P_df = pd.DataFrame(P, index=states, columns=states)
    print(P_df)

    # 5. Save the transition matrix to CSV
    P_df.to_csv(OUTPUT_TRANSITION_MATRIX)
    print(f"\nSaved transition matrix to {OUTPUT_TRANSITION_MATRIX}")


if __name__ == "__main__":
    main()
