import argparse
import base64
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser("Encode an image file as base64 text")
    parser.add_argument("--input", required=True, type=str)
    parser.add_argument("--output", default="", type=str, help="Optional output text file path")
    parser.add_argument("--wrap", default=0, type=int, help="Wrap base64 text every N chars; 0 means no wrap")
    return parser.parse_args()


def wrap_text(text, width):
    if width <= 0:
        return text
    return "\n".join(text[i : i + width] for i in range(0, len(text), width))


def main():
    args = parse_args()
    input_path = Path(args.input)
    if not input_path.is_file():
        raise FileNotFoundError(f"Input image not found: {input_path}")

    encoded = base64.b64encode(input_path.read_bytes()).decode("ascii")
    encoded = wrap_text(encoded, args.wrap)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(encoded)
        print("saved base64 to:", output_path.resolve())
    else:
        print(encoded)


if __name__ == "__main__":
    main()
