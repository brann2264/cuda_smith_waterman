import random
import sys

# --- Configuration ---
NUM_PAIRS = 1      # How many sequence pairs to generate
D_MIN, D_MAX = 50000, 150000  # Length range for Reference (D)
Q_MIN, Q_MAX = 100, 300   # Length range for Query (Q)

nucleotides = ['A', 'C', 'G', 'T', 'N']
weights = [10, 10, 10, 10, 1] # Make 'N' slightly less common

filename = "datasets/generated_data.txt"

print(f"Generating {NUM_PAIRS} sequence pairs...")

with open(filename, "w") as f:
    for i in range(NUM_PAIRS):
        # 1. Generate random Reference (D)
        d_len = random.randint(D_MIN, D_MAX)
        d_seq = "".join(random.choices(nucleotides, weights=weights, k=d_len))
        
        # 2. Extract a slice to be the Query (Q)
        q_len = random.randint(Q_MIN, min(Q_MAX, d_len))
        start_idx = random.randint(0, d_len - q_len)
        q_seq = list(d_seq[start_idx : start_idx + q_len])
        
        # 3. Introduce mutations (10% substitution rate)
        for j in range(len(q_seq)):
            if random.random() < 0.10:
                q_seq[j] = random.choice(nucleotides)
                
        q_seq = "".join(q_seq)
        
        # 4. Write in your exact required format
        f.write(f"Q:\t{q_seq}\n")
        f.write(f"D:\t{d_seq}\n")

print(f"Success! Saved to '{filename}'")