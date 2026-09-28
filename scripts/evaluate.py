"""Evaluate PSNR, SSIM, D-SSIM, MS-SSIM, and VGG LPIPS.

Two input modes share the metric implementations in
``gaussian_splatting.evaluation``, so they agree on identical images:

* checkpoint mode (``--checkpoint``): render train/val/test views of a
  checkpoint and write per-image PSNR / SSIM / LPIPS to a JSON file.
* image mode (``--render-dir``): pair rendered images already on disk with
  ground-truth images, score them with the metric set of ``--dataset``, and
  write ``metrics.csv`` / ``summary.json`` / ``per_frame.csv``.

Examples::

    python scripts/evaluate.py --data data/nerf_synthetic/lego \\
        --checkpoint output/.../checkpoints/iteration_00030000.pt \\
        --split test --output output/.../evaluation.json

    python scripts/evaluate.py --dataset neu3d \\
        --render-dir output/.../renders/renders --gt-dir output/.../renders/gt \\
        --output-csv output/.../results/metrics.csv \\
        --json-out output/.../results/summary.json \\
        --per-frame-csv output/.../results/per_frame.csv \\
        --rgba-background black
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from gaussian_splatting.config import config_from_mapping, resolve_device, resolve_dtype
from gaussian_splatting.data import load_dataset
from gaussian_splatting.evaluation import images
from gaussian_splatting.evaluation.runner import evaluate_camera_set
from gaussian_splatting.io.checkpoint import model_from_checkpoint_state, read_checkpoint
from gaussian_splatting.renderer import GaussianRenderer

ARROWS = {"psnr": "↑", "ssim": "↑", "ms_ssim": "↑", "d_ssim": "↓", "lpips": "↓"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    checkpoint = parser.add_argument_group("checkpoint mode")
    checkpoint.add_argument("--checkpoint", type=Path)
    checkpoint.add_argument("--data", type=Path)
    checkpoint.add_argument("--split", choices=("train", "val", "test"))
    checkpoint.add_argument("--output", type=Path, help="evaluation JSON path")
    checkpoint.add_argument("--image-directory", default="images")
    checkpoint.add_argument(
        "--render-backend",
        choices=("reference", "cuda"),
        default="reference",
    )

    image = parser.add_argument_group("image mode")
    image.add_argument("--render-dir", type=Path)
    image.add_argument("--gt-dir", type=Path)
    image.add_argument("--dataset", choices=sorted(images.DATASET_METRICS))
    image.add_argument("--output-csv", type=Path, help="per-image CSV (metrics.csv)")
    image.add_argument(
        "--json-out",
        type=Path,
        help="per-camera summary JSON; written only when files are named per camera",
    )
    image.add_argument(
        "--per-frame-csv",
        type=Path,
        help="frame,camera,<metrics> CSV; frames come from frame_NNNN/ directories",
    )
    image.add_argument(
        "--block",
        type=int,
        default=10,
        help="frame block width for the printed per-block trend",
    )
    image.add_argument(
        "--rgba-background",
        choices=("white", "black"),
        default="white",
        help="RGBA compositing background; match data.rgba_background of the run",
    )
    image.add_argument("--device", default="cuda")
    return parser


def evaluate_checkpoint(args: argparse.Namespace) -> int:
    state = read_checkpoint(args.checkpoint, map_location="cpu")
    config = config_from_mapping(state["config"])
    device = resolve_device(config.runtime)
    dtype = resolve_dtype(config.runtime)
    model = model_from_checkpoint_state(state, config, device=device, dtype=dtype)
    dataset = load_dataset(
        args.data,
        config,
        splits=(args.split,),
        load_points=False,
        image_directory=args.image_directory,
    )
    selected = getattr(dataset, args.split)
    evaluate_camera_set(
        model,
        GaussianRenderer(config.rendering, backend=args.render_backend),
        selected,
        args.output,
        ssim_window_size=config.loss.ssim_window_size,
        ssim_sigma=config.loss.ssim_sigma,
        ssim_k1=config.loss.ssim_k1,
        ssim_k2=config.loss.ssim_k2,
    )
    return 0


def _print_table(
    title: str,
    names: tuple[str, ...],
    entries: list[tuple[str, int, dict[str, float]]],
) -> None:
    print(f"\n{'─' * 60}\n  {title}\n{'─' * 60}")
    print(f"  {'':<14}{'n':>7}" + "".join(f"{name + ' ' + ARROWS[name]:>14}" for name in names))
    for label, count, values in entries:
        cells = "".join(f"{values[name]:>14.4f}" for name in names)
        print(f"  {label:<14}{count:>7}{cells}")


def evaluate_images(args: argparse.Namespace) -> int:
    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("[警告] CUDA 利用不可。CPU にフォールバック。")
        device = "cpu"

    found = images.collect_image_pairs(args.render_dir, args.gt_dir)
    if found.missing:
        print(f"[警告] GT が見つからないファイル ({len(found.missing)} 件): {found.missing[:5]}")
    if found.ambiguous:
        print(
            f"[警告] GT が一意に定まらないファイル ({len(found.ambiguous)} 件、スキップ): "
            f"{found.ambiguous[:5]}"
        )
    if not found.pairs:
        print("[エラー] 有効な画像ペアがありません")
        return 1
    print(f"dataset: {args.dataset} / 画像ペア: {len(found.pairs):,} 組")

    metrics = images.ImageMetrics(args.dataset, device=device)
    names = metrics.names
    rows = images.evaluate_image_pairs(
        found.pairs,
        metrics,
        render_dir=args.render_dir,
        background=args.rgba_background,
        progress_every=100,
    )

    _print_table("全体", names, [("全体", len(rows), images.metric_averages(rows, names))])

    if args.output_csv:
        images.write_metrics_csv(rows, names, args.output_csv)
        print(f"CSV: {args.output_csv}")
    if args.per_frame_csv:
        images.write_per_frame_csv(rows, names, args.per_frame_csv)
        print(f"フレーム別 CSV: {args.per_frame_csv}")

    summary = images.camera_summary(rows, names)
    if summary is not None:
        cameras = [
            (name, int(entry["count"]), {m: entry[m]["mean"] for m in names if m in entry})
            for name, entry in summary["cameras"].items()
        ]
        _print_table("カメラ別 (per-image の平均)", names, cameras)
        if args.json_out:
            images.write_camera_summary_json(summary, args.json_out, metrics_csv=args.output_csv)
            print(f"JSON: {args.json_out}")
    elif args.json_out:
        print("[スキップ] ファイル名がカメラ別に分かれていないため JSON は出さない")

    blocks = images.frame_blocks(rows, names, args.block)
    if blocks:
        _print_table(f"{args.block} フレームブロックごとの推移", names, blocks)
        if len(blocks) > 1 and "psnr" in names:
            first, last = blocks[0], blocks[-1]
            best = max(blocks, key=lambda block: block[2]["psnr"])
            worst = min(blocks, key=lambda block: block[2]["psnr"])
            print(f"  PSNR 末尾 {last[0]} - 先頭 {first[0]}: "
                  f"{last[2]['psnr'] - first[2]['psnr']:+.4f} dB")
            print(f"  PSNR 最良 {best[0]} {best[2]['psnr']:.4f} dB / "
                  f"最低 {worst[0]} {worst[2]['psnr']:.4f} dB "
                  f"(振れ幅 {best[2]['psnr'] - worst[2]['psnr']:.4f} dB)")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if (args.checkpoint is None) == (args.render_dir is None):
        parser.error("specify exactly one of --checkpoint or --render-dir")
    if args.checkpoint is not None:
        missing = [flag for flag, value in (("--data", args.data), ("--split", args.split),
                                            ("--output", args.output)) if value is None]
        if missing:
            parser.error(f"checkpoint mode requires {', '.join(missing)}")
        return evaluate_checkpoint(args)
    missing = [flag for flag, value in (("--gt-dir", args.gt_dir), ("--dataset", args.dataset))
               if value is None]
    if missing:
        parser.error(f"image mode requires {', '.join(missing)}")
    return evaluate_images(args)


if __name__ == "__main__":
    raise SystemExit(main())
