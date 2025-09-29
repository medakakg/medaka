"""
This script filters biomedical triples from an input CSV file
based on the `Confidence` column and saves only those
with confidence scores greater than or equal to a threshold.

CLI Arguments:
    --input-csv     Path to the input CSV file (required)
    --output-csv    Path to save the filtered CSV file (required)
    --threshold     Confidence threshold (default=0.5)
"""

# === Imports ===
import pandas as pd
import argparse

# === Helper Function ===
def filter_triples(input_csv, output_csv, threshold=0.5):
    """
    Filters triples from input_csv based on Confidence column
    and writes only those with confidence >= threshold to output_csv.
    """
    df = pd.read_csv(input_csv)
    df['Confidence'] = pd.to_numeric(df['Confidence'], errors='coerce')
    filtered_df = df[df['Confidence'] >= threshold]
    filtered_df.to_csv(output_csv, index=False)
    print(f"Filtered triples saved to {output_csv}. "
          f"{len(filtered_df)} triples kept out of {len(df)}.")

# === CLI Parser ===
def parse_args():
    parser = argparse.ArgumentParser(description="Filter triples based on confidence scores.")
    parser.add_argument("--input-csv", required=True, help="Path to the input CSV file.")
    parser.add_argument("--output-csv", required=True, help="Path to save the filtered CSV file.")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Confidence threshold (default: 0.5).")
    return parser.parse_args()

# === Entry Point ===
if __name__ == "__main__":
    args = parse_args()
    filter_triples(args.input_csv, args.output_csv, threshold=args.threshold)
