"""
eval/benchmark_fps.py
学習済みチェックポイントのレンダリング速度 (FPS) を測る。

使い方:
    python eval/benchmark_fps.py \
        --run_dir output/4DGS/neu3d/coffee_martini/baseline_30k \
        --data_dir data/neu3d/coffee_martini/converted_4d \
        --frames 1 50 100 150 200 250 300

----------------------------------------------------------------------------
何を測り、何を測らないか
----------------------------------------------------------------------------
測るのは **ラスタライズ 1 回の所要時間** だけ。チェックポイントの読み込み、
データセットの構築、PNG エンコードは含めない。これらは推論のたびに起きる
処理ではないので、混ぜると FPS が実態より大幅に低く出る。

GPU は非同期に走るので、``torch.cuda.synchronize()` を挟まずに時刻を取ると
カーネル投入時刻を測ってしまい、ありえない FPS が出る。計測の前後で必ず同期する。

最初の数回はカーネルの JIT/キャッシュで遅いため warmup として捨てる。
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
for root in (REPO_ROOT / "src",):
    if root.is_dir() and str(root) not in sys.path:
        sys.path.insert(0, str(root))

from gaussian_splatting.config import (  # noqa: E402
    config_from_mapping,
    resolve_device,
    resolve_dtype,
)
from gaussian_splatting.data import load_dataset  # noqa: E402
from gaussian_splatting.io.checkpoint import (  # noqa: E402
    model_from_checkpoint_state,
    read_checkpoint,
)
from gaussian_splatting.renderer import GaussianRenderer  # noqa: E402
from gaussian_splatting.training.trainer import camera_to  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--run_dir", type=Path, required=True)
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument("--frames", type=int, nargs="+", required=True)
    parser.add_argument("--split", default="test", choices=("train", "test", "val"))
    parser.add_argument("--image-directory", default="images")
    parser.add_argument("--render-backend", choices=("reference", "cuda"), default="cuda")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10,
                        help="カメラ 1 台あたりの計測回数 (warmup を除く)")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    per_frame: list[tuple[int, float, int]] = []
    all_times: list[float] = []

    for number in args.frames:
        checkpoint = args.run_dir / f"frame_{number:04d}" / "checkpoints" / "latest.pt"
        if not checkpoint.is_file():
            print(f"[スキップ] frame {number}: {checkpoint} がありません")
            continue

        state = read_checkpoint(checkpoint, map_location="cpu")
        config = config_from_mapping(state["config"])
        device = resolve_device(config.runtime)
        dtype = resolve_dtype(config.runtime)
        model = model_from_checkpoint_state(state, config, device=device, dtype=dtype)
        dataset = load_dataset(
            args.data_dir / f"frame_{number:04d}", config,
            splits=(args.split,), load_points=False,
            image_directory=args.image_directory,
        )
        cameras = getattr(dataset, args.split)
        renderer = GaussianRenderer(config.rendering, backend=args.render_backend)

        count = int(model.means_world.shape[0])
        times: list[float] = []
        with torch.no_grad():
            for source_camera in cameras:
                # カメラを GPU 側へ移すのは 1 台につき 1 回。GT 画像は
                # レンダリングに要らないので載せない (転送分を測りたくない)。
                camera = camera_to(
                    source_camera,
                    device=model.means_world.device,
                    dtype=model.means_world.dtype,
                    include_image=False,
                )
                for index in range(args.warmup + args.repeats):
                    if device.type == "cuda":
                        torch.cuda.synchronize()
                    started = time.perf_counter()
                    renderer(model, camera)
                    if device.type == "cuda":
                        torch.cuda.synchronize()
                    elapsed = time.perf_counter() - started
                    if index >= args.warmup:
                        times.append(elapsed)

        median = statistics.median(times)
        per_frame.append((number, median, count))
        all_times.extend(times)
        print(f"  frame {number:04d}: {1.0 / median:6.2f} FPS "
              f"({median * 1000:6.1f} ms/枚, {count:,} Gaussians)")

        del model, dataset, cameras, state
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if not all_times:
        print("[エラー] 計測できたフレームがありません")
        return 1

    overall = statistics.median(all_times)
    print()
    print(f"  中央値    : {1.0 / overall:.2f} FPS ({overall * 1000:.1f} ms/枚)")
    fastest = max(per_frame, key=lambda r: 1.0 / r[1])
    slowest = min(per_frame, key=lambda r: 1.0 / r[1])
    print(f"  最速      : {1.0 / fastest[1]:.2f} FPS (frame {fastest[0]:04d}, "
          f"{fastest[2]:,} Gaussians)")
    print(f"  最遅      : {1.0 / slowest[1]:.2f} FPS (frame {slowest[0]:04d}, "
          f"{slowest[2]:,} Gaussians)")
    print(f"  計測回数  : {len(all_times)} 回 "
          f"(warmup {args.warmup} 回/台を除外, backend={args.render_backend})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
