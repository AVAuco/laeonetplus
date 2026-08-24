"""Command-line entry point for the LAEO-Net++ pipeline."""

from __future__ import annotations

import argparse

from .pipeline import LaeoPipeline


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Detect heads, track them, and estimate who is Looking At Each Other (LAEO)."
    )
    parser.add_argument("--video", required=True, help="Path to the input video file")
    parser.add_argument("--output_video", default=None, help="Path to write the annotated output video (optional)")
    parser.add_argument("--output_json", default=None, help="Path to write the per-frame LAEO probabilities as JSON")
    parser.add_argument("--threshold", type=float, default=0.5, help="LAEO probability threshold for green/red coloring")

    parser.add_argument("--face_model", default="yolov8n-face.pt", help="Path to YOLOv8-face weights")
    parser.add_argument("--detector_conf", type=float, default=0.35, help="Face detector confidence threshold")
    parser.add_argument("--head_scale", type=float, default=1.8, help="Face-to-head bbox expansion factor")

    parser.add_argument("--laeo_model", choices=["ava", "uco"], default="ava", help="Which pretrained LAEO-Net++ model to use")
    parser.add_argument("--laeo_json", default=None, help="Override: path to the LAEO model config JSON")
    parser.add_argument("--laeo_weights", default=None, help="Override: path to the LAEO model weights HDF5")
    parser.add_argument("--mean_map", default="models/meanmaps10.npy", help="Path to the mean head-map .npy file")

    parser.add_argument("--device", default="cpu", help="Torch device, e.g. 'cpu' or 'cuda:0'")
    parser.add_argument("--max_age", type=int, default=30, help="DeepSORT: frames to keep a track alive without a match")
    parser.add_argument("--n_init", type=int, default=3, help="DeepSORT: frames needed to confirm a new track")
    parser.add_argument("--max_frames", type=int, default=None, help="Optional cap on the number of frames to process")

    return parser


def main(argv=None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    laeo_json = args.laeo_json or f"models_pt/laeonet_{args.laeo_model}.json"
    laeo_weights = args.laeo_weights or f"models_pt/laeonet_{args.laeo_model}_weights.h5"

    pipeline = LaeoPipeline(
        face_weights=args.face_model,
        laeo_json=laeo_json,
        laeo_weights=laeo_weights,
        mean_map_path=args.mean_map,
        device=args.device,
        detector_conf=args.detector_conf,
        head_scale=args.head_scale,
        max_age=args.max_age,
        n_init=args.n_init,
    )

    result = pipeline.run(
        video_path=args.video,
        output_video=args.output_video,
        output_json=args.output_json,
        threshold=args.threshold,
        max_frames=args.max_frames,
    )

    n_pairs = len(result["pairs"])
    print(f"Processed {result['num_frames']} frames, found {n_pairs} LAEO pair(s).")
    if args.output_video:
        print(f"Annotated video written to {args.output_video}")
    if args.output_json:
        print(f"Per-frame LAEO probabilities written to {args.output_json}")


if __name__ == "__main__":
    main()
