import random
import argparse
import os

def main():
    # Set up the command line argument parser
    parser = argparse.ArgumentParser(description="Generate DNA sequence pairs for Smith-Waterman testing.")
    
    # Define the arguments
    parser.add_argument("-n", "--num_pairs", type=int, default=1, 
                        help="Number of sequence pairs to generate (default: 1)")
    parser.add_argument("-d", "--d_len", type=int, default=1000000, 
                        help="Length of the Reference/Database sequence (default: 1,000,000)")
    parser.add_argument("-q", "--q_len", type=int, default=300, 
                        help="Length of the Query sequence (default: 300)")
    parser.add_argument("-o", "--output", type=str, default="datasets/generated_data.txt", 
                        help="Output filename (default: datasets/generated_data.txt)")

    # Parse the arguments provided by the user
    args = parser.parse_args()

    # Create the output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else '.', exist_ok=True)

    nucleotides = ['A', 'C', 'G', 'T', 'N']
    weights = [10, 10, 10, 10, 1]

    print(f"Generating {args.num_pairs} sequence pair(s)...")
    print(f" -> D Length: {args.d_len}")
    print(f" -> Q Length: {args.q_len}")

    with open(args.output, "w") as f:
        for i in range(args.num_pairs):
            # 1. Generate random Reference (D)
            d_seq = "".join(random.choices(nucleotides, weights=weights, k=args.d_len))
            
            # 2. Extract a slice to be the Query (Q)
            # Ensure Q isn't accidentally longer than D
            actual_q_len = min(args.q_len, args.d_len)
            start_idx = random.randint(0, args.d_len - actual_q_len)
            q_seq = list(d_seq[start_idx : start_idx + actual_q_len])
            
            # 3. Introduce mutations (10% substitution rate)
            for j in range(len(q_seq)):
                if random.random() < 0.10:
                    q_seq[j] = random.choice(nucleotides)
                    
            q_seq = "".join(q_seq)
            
            # 4. Write in your exact required format
            f.write(f"Q:\t{q_seq}\n")
            f.write(f"D:\t{d_seq}\n")

    print(f"Success! Saved to '{args.output}'")

if __name__ == "__main__":
    main()