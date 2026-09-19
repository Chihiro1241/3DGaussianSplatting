"""
4DGS Evaluation Script
論文 (Wu et al., 2024) に準拠した評価スクリプト

対応データセット:
  - dnerf          (合成/4D):   PSNR / SSIM / LPIPS
  - hypernerf      (実世界/4D): PSNR / MS-SSIM
  - neu3d          (実世界/4D): PSNR / D-SSIM / LPIPS
  - nerf_synthetic (合成/静的): PSNR / SSIM / LPIPS   ... 本リポジトリの data/static/nerf_synthetic
  - colmap         (実世界/静的): PSNR / SSIM / LPIPS ... data/static/mipnerf360, data/static/tandt_db

このスクリプトは「レンダリング済み PNG と GT PNG のディレクトリ対」を突き合わせる
オフライン評価器である。チェックポイントから直接評価したい場合は、リポジトリ本体の
``scripts/evaluate.py`` (gaussian_splatting.evaluation.runner) を使うこと。

レンダリング画像は ``scripts/render.py`` が出力する **フラットなベース名**
(``<stem>.png``) を前提とする。GT 側がサブディレクトリ構造を持つ場合
(nerf_synthetic の ``test/``) でもベース名で解決できるよう、相対パス一致に失敗した
ものはベース名でフォールバック照合する。

使い方:
  python eval/evaluate.py \
      --dataset nerf_synthetic \
      --render_dir /path/to/rendered_images \
      --gt_dir    /path/to/ground_truth_images \
      [--output_csv results.csv]
"""

import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image


def _import_torch():
    try:
        import torch
        return torch
    except ImportError:
        raise ImportError("PyTorch が見つかりません。`pip install torch` を実行してください。")


def _import_metrics():
    torch = _import_torch()
    try:
        import torchmetrics.image  # noqa: F401
        USE_TORCHMETRICS = True
    except ImportError:
        USE_TORCHMETRICS = False
    return torch, USE_TORCHMETRICS


def load_image_np(path: Path, rgba_background: str = "white") -> np.ndarray:
    """RGB の uint8 配列を返す。

    RGBA 画像は **必ず背景へアルファ合成する**。``convert("RGB")`` は
    アルファを単に捨てるため、透明画素に RGB=(0,0,0) を格納している
    D-NeRF / NeRF Synthetic の GT では背景が黒になり、白背景で
    レンダリングした画像と比較すると PSNR が 1 dB 台まで落ちる。
    本体の ``gaussian_splatting.data.dataset._image_to_tensor`` と同じ規則。
    """
    img = Image.open(path)
    if img.mode == "RGBA":
        rgba = np.asarray(img, dtype=np.float32)
        alpha = rgba[..., 3:4] / 255.0
        background = 0.0 if rgba_background == "black" else 255.0
        rgb = rgba[..., :3] * alpha + background * (1.0 - alpha)
        return np.clip(np.rint(rgb), 0, 255).astype(np.uint8)
    return np.array(img.convert("RGB"), dtype=np.uint8)


def np_to_tensor(img_np: np.ndarray, device="cpu"):
    torch = _import_torch()
    t = torch.from_numpy(img_np).float() / 255.0
    t = t.permute(2, 0, 1).unsqueeze(0)
    return t.to(device)


class MetricsCalculator:
    DATASET_METRICS = {
        # --- 論文 (Wu et al., 2024) の 4D ベンチマーク ---
        "dnerf":          ["psnr", "ssim", "lpips"],
        "hypernerf":      ["psnr", "ms_ssim"],
        "neu3d":          ["psnr", "d_ssim", "lpips"],
        # --- 本リポジトリに実在する静的データセット (3DGS baseline) ---
        "nerf_synthetic": ["psnr", "ssim", "lpips"],
        "colmap":         ["psnr", "ssim", "lpips"],
    }

    def __init__(self, dataset_type: str, device: str = "cpu", lpips_net: str = "vgg"):
        self.dataset_type = dataset_type.lower()
        assert self.dataset_type in self.DATASET_METRICS, \
            f"dataset_type は {list(self.DATASET_METRICS)} のいずれかにしてください。"
        self.metrics_needed = self.DATASET_METRICS[self.dataset_type]
        self.device = device
        # 本体の LPIPSMetric (gaussian_splatting/evaluation/metrics.py) は VGG を使う。
        # 既存の scripts/evaluate.py と数値を揃えるため既定は "vgg"。
        self.lpips_net = lpips_net
        torch, USE_TORCHMETRICS = _import_metrics()
        self.torch = torch
        self.USE_TORCHMETRICS = USE_TORCHMETRICS
        if USE_TORCHMETRICS:
            self._init_torchmetrics()
        else:
            print("[警告] torchmetrics が見つかりません。skimage / numpy フォールバックを使用します。")
            self._init_fallback()

    def _init_torchmetrics(self):
        from torchmetrics.image import (
            PeakSignalNoiseRatio,
            StructuralSimilarityIndexMeasure,
            MultiScaleStructuralSimilarityIndexMeasure,
            LearnedPerceptualImagePatchSimilarity,
        )
        self._stateful = []
        if "psnr" in self.metrics_needed:
            self.psnr_fn = PeakSignalNoiseRatio(data_range=1.0).to(self.device)
            self._stateful.append(self.psnr_fn)
        if "ssim" in self.metrics_needed or "d_ssim" in self.metrics_needed:
            self.ssim_fn = StructuralSimilarityIndexMeasure(data_range=1.0).to(self.device)
            self._stateful.append(self.ssim_fn)
        if "ms_ssim" in self.metrics_needed:
            self.ms_ssim_fn = MultiScaleStructuralSimilarityIndexMeasure(data_range=1.0).to(self.device)
            self._stateful.append(self.ms_ssim_fn)
        if "lpips" in self.metrics_needed:
            self.lpips_fn = LearnedPerceptualImagePatchSimilarity(
                net_type=self.lpips_net
            ).to(self.device)
            self._stateful.append(self.lpips_fn)

    def _init_fallback(self):
        try:
            from skimage.metrics import peak_signal_noise_ratio, structural_similarity
            self._skimage_psnr = peak_signal_noise_ratio
            self._skimage_ssim = structural_similarity
        except ImportError:
            raise ImportError(
                "torchmetrics も scikit-image も見つかりません。\n"
                "  pip install -e '.[eval]'\n"
                "  または pip install 'torchmetrics[image]' scikit-image\n"
                "のいずれかを実行してください。"
            )

    def compute(self, pred_np: np.ndarray, gt_np: np.ndarray) -> dict:
        results = {}
        if self.USE_TORCHMETRICS:
            pred_t = np_to_tensor(pred_np, self.device)
            gt_t   = np_to_tensor(gt_np,   self.device)
            with self.torch.no_grad():
                if "psnr" in self.metrics_needed:
                    results["psnr"] = float(self.psnr_fn(pred_t, gt_t))
                if "ssim" in self.metrics_needed:
                    results["ssim"] = float(self.ssim_fn(pred_t, gt_t))
                if "d_ssim" in self.metrics_needed:
                    results["d_ssim"] = 1.0 - float(self.ssim_fn(pred_t, gt_t))
                if "ms_ssim" in self.metrics_needed:
                    results["ms_ssim"] = float(self.ms_ssim_fn(pred_t, gt_t))
                if "lpips" in self.metrics_needed:
                    pred_lpips = (pred_t * 2.0 - 1.0).clamp(-1.0, 1.0)
                    gt_lpips   = (gt_t   * 2.0 - 1.0).clamp(-1.0, 1.0)
                    results["lpips"] = float(self.lpips_fn(pred_lpips, gt_lpips))
            # torchmetrics の Metric は呼び出しごとに内部状態を蓄積するため、
            # 数百枚を回すと際限なくメモリを消費する。1 枚ごとに破棄する。
            for metric in self._stateful:
                metric.reset()
        else:
            pred_f = pred_np.astype(np.float64) / 255.0
            gt_f   = gt_np.astype(np.float64)   / 255.0
            if "psnr" in self.metrics_needed:
                results["psnr"] = float(self._skimage_psnr(gt_f, pred_f, data_range=1.0))
            if "ssim" in self.metrics_needed or "d_ssim" in self.metrics_needed:
                ssim_val = float(self._skimage_ssim(gt_f, pred_f, channel_axis=-1, data_range=1.0))
                if "ssim"   in self.metrics_needed: results["ssim"]   = ssim_val
                if "d_ssim" in self.metrics_needed: results["d_ssim"] = 1.0 - ssim_val
            if "ms_ssim" in self.metrics_needed:
                print("[警告] skimage は MS-SSIM をサポートしていません。")
                results["ms_ssim"] = float("nan")
            if "lpips" in self.metrics_needed:
                print("[警告] skimage フォールバックでは LPIPS を計算できません。")
                results["lpips"] = float("nan")
        return results


SUPPORTED_EXT = {".png", ".jpg", ".jpeg", ".exr"}

# NeRF Synthetic の test/ には GT の RGB と一緒に深度・法線が入っている。
# GT 側を走査する場面でこれらを拾わないよう除外する。
GT_EXCLUDE_SUBSTRINGS = ("_depth_", "_normal_")


def _is_gt_rgb(path: Path) -> bool:
    return not any(token in path.name for token in GT_EXCLUDE_SUBSTRINGS)


def collect_image_pairs(render_dir: Path, gt_dir: Path):
    """レンダリング画像と GT 画像を対応付ける。

    1. 相対パス一致 (render_dir/a/b.png <-> gt_dir/a/b.png) を最優先で試す。
    2. 失敗した場合、GT 側をベース名で索引して照合する。
       ``scripts/render.py`` は image_name のベース名だけをフラットに書き出すため
       (evaluation/runner.py の render_camera_set)、GT が ``<scene>/test/r_0.png``
       のようにサブディレクトリを持つケースはこちらで解決される。
    """
    if not render_dir.is_dir():
        raise FileNotFoundError(f"レンダリングディレクトリが存在しません: {render_dir}")
    if not gt_dir.is_dir():
        raise FileNotFoundError(f"GT ディレクトリが存在しません: {gt_dir}")

    render_files = sorted(
        p for p in render_dir.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_EXT
    )

    gt_by_stem: dict[str, list[Path]] = {}
    for g in gt_dir.rglob("*"):
        if g.is_file() and g.suffix.lower() in SUPPORTED_EXT and _is_gt_rgb(g):
            gt_by_stem.setdefault(g.stem, []).append(g)

    pairs, missing, ambiguous = [], [], []
    for r in render_files:
        rel = r.relative_to(render_dir)
        direct = gt_dir / rel
        if direct.is_file():
            pairs.append((r, direct))
            continue
        candidates = gt_by_stem.get(r.stem, [])
        if len(candidates) == 1:
            pairs.append((r, candidates[0]))
        elif len(candidates) > 1:
            ambiguous.append(str(rel))
        else:
            missing.append(str(rel))

    if missing:
        print(f"[警告] GT が見つからないファイル ({len(missing)} 件): {missing[:5]}")
    if ambiguous:
        print(
            f"[警告] GT が一意に定まらないファイル ({len(ambiguous)} 件、スキップ): "
            f"{ambiguous[:5]}"
        )
    return pairs


def get_model_size_mb(model_path: str) -> float:
    p = Path(model_path)
    if not p.exists(): return float("nan")
    if p.is_file(): return p.stat().st_size / (1024 ** 2)
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) / (1024 ** 2)


def _finite_mean(values) -> float:
    finite = [v for v in values if np.isfinite(v)]
    if not finite:
        return float("nan")
    return float(np.mean(finite))


def run_evaluation(dataset_type, render_dir, gt_dir, output_csv=None, model_path=None,
                   device="cpu", lpips_net="vgg", rgba_background="white"):
    print(f"\n{'='*60}\n  4DGS 評価  |  dataset: {dataset_type.upper()}\n{'='*60}")
    pairs = collect_image_pairs(Path(render_dir), Path(gt_dir))
    if not pairs:
        raise FileNotFoundError("有効な画像ペアが見つかりませんでした。")
    print(f"  画像ペア数: {len(pairs)}")

    calc = MetricsCalculator(dataset_type=dataset_type, device=device, lpips_net=lpips_net)
    metric_names = calc.metrics_needed
    all_results = []

    for i, (pred_path, gt_path) in enumerate(pairs):
        pred_np = load_image_np(pred_path, rgba_background)
        gt_np   = load_image_np(gt_path,   rgba_background)
        if pred_np.shape != gt_np.shape:
            h, w = gt_np.shape[:2]
            pred_np = np.array(Image.fromarray(pred_np).resize((w, h), Image.LANCZOS))
        metrics = calc.compute(pred_np, gt_np)
        metrics["filename"] = pred_path.name
        all_results.append(metrics)
        if (i + 1) % 50 == 0 or (i + 1) == len(pairs):
            print(f"  処理中: {i+1}/{len(pairs)}")

    # PSNR は完全一致で +inf になり得るので、非有限値は平均から除外する。
    averages = {
        m: _finite_mean([r.get(m, float("nan")) for r in all_results])
        for m in metric_names
    }
    if model_path:
        averages["storage_mb"] = get_model_size_mb(model_path)

    DISPLAY = {"psnr": ("PSNR (dB)", "↑"), "ssim": ("SSIM", "↑"), "ms_ssim": ("MS-SSIM", "↑"),
               "d_ssim": ("D-SSIM", "↓"), "lpips": ("LPIPS", "↓"), "storage_mb": ("Storage (MB)", "↓")}
    print(f"\n{'─'*40}\n  評価結果 ({dataset_type.upper()})\n{'─'*40}")
    for k, v in averages.items():
        label, arrow = DISPLAY.get(k, (k, ""))
        print(f"  {label:20s} {arrow}  {v:.4f}")

    if output_csv:
        destination = Path(output_csv)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with open(destination, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["filename"] + metric_names, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(all_results)
            writer.writerow({"filename": "*** AVERAGE ***", **{m: f"{averages[m]:.4f}" for m in metric_names}})
        print(f"  CSV 保存: {destination}")
    return averages


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset",    required=True,
                   choices=sorted(MetricsCalculator.DATASET_METRICS))
    p.add_argument("--render_dir", required=True)
    p.add_argument("--gt_dir",     required=True)
    p.add_argument("--output_csv", default=None)
    p.add_argument("--model_path", default=None)
    p.add_argument("--device",     default="cuda")
    p.add_argument("--lpips_net",  default="vgg", choices=["vgg", "alex", "squeeze"],
                   help="LPIPS backbone (既定 vgg: 本体 evaluation/metrics.py と同一)")
    p.add_argument("--rgba_background", default="white", choices=["white", "black"],
                   help="RGBA 画像の合成背景。学習に使った config の data.rgba_background "
                        "および rendering.background と一致させること (既定 white)")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    try:
        import torch
        if args.device == "cuda" and not torch.cuda.is_available():
            print("[警告] CUDA 利用不可。CPU にフォールバック。")
            args.device = "cpu"
    except ImportError:
        args.device = "cpu"

    run_evaluation(args.dataset, args.render_dir, args.gt_dir,
                   args.output_csv, args.model_path, args.device, args.lpips_net,
                   args.rgba_background)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
