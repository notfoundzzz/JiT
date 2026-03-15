import argparse
from pathlib import Path

import requests


IMAGE_URLS = {
    "kodim03.png": "https://r0k.us/graphics/kodak/kodak/kodim03.png",
    "kodim12.png": "https://r0k.us/graphics/kodak/kodak/kodim12.png",
    "kodim19.png": "https://r0k.us/graphics/kodak/kodak/kodim19.png",
}


def parse_args():
    parser = argparse.ArgumentParser("Download a few natural images for black-box JiT testing")
    parser.add_argument("--output_dir", default="./test_data/kodak", type=str)
    return parser.parse_args()


def download_file(url, output_path):
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    output_path.write_bytes(response.content)


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for name, url in IMAGE_URLS.items():
        output_path = output_dir / name
        if output_path.exists():
            print(f"skip {name}")
            continue
        print(f"download {name}")
        download_file(url, output_path)

    print(f"ready: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
