"""Write minimal submission CSV with header buyer_id,predicted_id (no path logic)."""
import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Write minimal submission CSV.")
    parser.add_argument("--output", required=True, help="Output CSV path.")
    args = parser.parse_args()
    with open(args.output, "w", encoding="utf-8") as f:
        f.write("buyer_id,predicted_id\n")


if __name__ == "__main__":
    main()
