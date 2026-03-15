import argparse
from pathlib import Path

import cv2
import requests


VIDEO_URLS = {
    "vtest.avi": "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/vtest.avi",
}


def parse_args():
    parser = argparse.ArgumentParser("Download a tiny video sample and extract a pair of frames")
    parser.add_argument("--output_dir", default="./test_data/video", type=str)
    parser.add_argument("--video_name", default="vtest.avi", choices=sorted(VIDEO_URLS.keys()))
    parser.add_argument("--frame0_idx", default=40, type=int)
    parser.add_argument("--frame1_idx", default=42, type=int)
    return parser.parse_args()


def download_file(url, output_path):
    response = requests.get(url, timeout=120)
    response.raise_for_status()
    output_path.write_bytes(response.content)


def save_frame(frame_bgr, output_path):
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    cv2.imwrite(str(output_path), cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))


def extract_frames(video_path, frame_ids, output_dir):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"failed to open video: {video_path}")

    requested = set(frame_ids)
    saved = set()
    frame_index = 0
    while requested - saved:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_index in requested:
            save_frame(frame, output_dir / f"frame_{frame_index:05d}.png")
            saved.add(frame_index)
        frame_index += 1
    cap.release()

    missing = requested - saved
    if missing:
        raise RuntimeError(f"missing frames: {sorted(missing)}")


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    video_path = output_dir / args.video_name
    if not video_path.exists():
        print(f"download {args.video_name}")
        download_file(VIDEO_URLS[args.video_name], video_path)
    else:
        print(f"skip {args.video_name}")

    frame_ids = [args.frame0_idx, args.frame1_idx]
    extract_frames(video_path, frame_ids, output_dir)
    print(f"ready: {output_dir.resolve()}")
    for frame_id in frame_ids:
        print(output_dir / f"frame_{frame_id:05d}.png")


if __name__ == "__main__":
    main()
