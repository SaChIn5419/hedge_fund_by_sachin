from __future__ import annotations

import argparse

from research.experiments.regime_validation import RegimeValidationReporter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Chimera regime validation report.")
    parser.add_argument("--trade-csv", required=True, help="Path to the Chimera trade log CSV")
    parser.add_argument("--output-prefix", required=True, help="Prefix for validation outputs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reporter = RegimeValidationReporter()
    out = reporter.generate(args.trade_csv, args.output_prefix)
    print(f"Wrote validation CSV: {out.csv_path}")
    print(f"Wrote validation TXT: {out.txt_path}")


if __name__ == "__main__":
    main()
