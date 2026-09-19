"""
4DGS 評価結果 集計スクリプト

eval/evaluate.py が出力した ``<dataset>_<scene>.csv`` を読み、データセット別に
シーン一覧・平均・論文値を並べて表示する。

使い方: python eval/summarize.py --results_dir ./eval/results
"""
import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

DATASET_CONFIG = {
    # --- 論文 (Wu et al., 2024) の 4D ベンチマーク ---
    "dnerf":     {"label": "D-NeRF (合成/4D) — Table 1",      "metrics": ["psnr","ssim","lpips"],    "resolution": "800×800"},
    "hypernerf": {"label": "HyperNeRF (実世界/4D) — Table 2",  "metrics": ["psnr","ms_ssim"],         "resolution": "960×540"},
    "neu3d":     {"label": "Neu3D (実世界/4D) — Table 3",      "metrics": ["psnr","d_ssim","lpips"],  "resolution": "1352×1014"},
    # --- 本リポジトリに実在する静的データセット (3DGS baseline) ---
    "nerf_synthetic": {"label": "NeRF Synthetic (合成/静的)", "metrics": ["psnr","ssim","lpips"], "resolution": "800×800"},
    "colmap":         {"label": "COLMAP 実写 (Mip-NeRF360 / T&T / DB)", "metrics": ["psnr","ssim","lpips"], "resolution": "scene 依存"},
}

# 表示順。長いキーから照合するため、この順序とは別に prefix 解決を行う。
DISPLAY_ORDER = ["dnerf", "hypernerf", "neu3d", "nerf_synthetic", "colmap"]

PAPER_REFERENCE = {
    "dnerf":     {"psnr": 34.05, "ssim": 0.98,  "lpips": 0.02},
    "hypernerf": {"psnr": 25.2,  "ms_ssim": 0.845},
    "neu3d":     {"psnr": 31.15, "d_ssim": 0.016, "lpips": 0.049},
    # nerf_synthetic / colmap は 4DGS 論文の対象外。3DGS 側の比較は
    # scripts/generate_paper_benchmark_report.py が manifest の paper_psnr で行う。
}
DIRECTIONS = {"psnr":"↑","ssim":"↑","ms_ssim":"↑","d_ssim":"↓","lpips":"↓"}


def parse_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if "AVERAGE" in (row.get("filename") or ""):
                return {
                    k: float(v)
                    for k, v in row.items()
                    if k != "filename" and v not in (None, "", "nan")
                }
    return {}


def split_stem(stem: str):
    """``<dataset>_<scene>`` を分解する。

    データセット名自体がアンダースコアを含む (``nerf_synthetic``) ため、
    単純な ``split("_", 1)`` では誤って ``nerf`` / ``synthetic_lego`` に割れる。
    既知のキーを長い順に照合して解決する。
    """
    for key in sorted(DATASET_CONFIG, key=len, reverse=True):
        prefix = key + "_"
        if stem.startswith(prefix) and len(stem) > len(prefix):
            return key, stem[len(prefix):]
    return None, None


def collect_results(results_dir):
    directory = Path(results_dir)
    if not directory.is_dir():
        raise FileNotFoundError(f"結果ディレクトリが存在しません: {directory}")
    data = defaultdict(dict)
    for p in sorted(directory.glob("*.csv")):
        dataset, scene = split_stem(p.stem)
        if dataset is None:
            continue
        m = parse_csv(p)
        if m:
            data[dataset][scene] = m
    return data


def print_table(dataset, scene_data):
    cfg = DATASET_CONFIG[dataset]
    metrics = cfg["metrics"]
    ref = PAPER_REFERENCE.get(dataset, {})
    print(f"\n{'═'*65}\n  {cfg['label']}  (解像度: {cfg['resolution']})\n{'═'*65}")
    print(f"  {'シーン':<22}" + "".join(f"  {m.upper():>10} {DIRECTIONS.get(m,'')}" for m in metrics))
    print(f"  {'─'*60}")
    scene_avgs = defaultdict(list)
    for scene, vals in sorted(scene_data.items()):
        print(f"  {scene:<22}" + "".join(f"  {vals.get(m, float('nan')):>12.4f}" for m in metrics))
        for m in metrics:
            scene_avgs[m].append(vals.get(m, float("nan")))
    print(f"  {'─'*60}")
    avgs = [
        float(np.nanmean(scene_avgs[m])) if np.any(np.isfinite(scene_avgs[m])) else float("nan")
        for m in metrics
    ]
    print(f"  {'平均 (Our Impl.)':<22}" + "".join(f"  {v:>12.4f}" for v in avgs))
    if ref:
        print(f"  {'論文値 (Wu+2024)':<22}" + "".join(f"  {ref.get(m, float('nan')):>12.4f}" for m in metrics))


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results_dir", required=True)
    args = p.parse_args()
    data = collect_results(args.results_dir)
    if not data:
        print(f"[エラー] 集計できる CSV が見つかりませんでした: {args.results_dir}")
        print(f"         ファイル名は '<dataset>_<scene>.csv' 形式で、dataset は "
              f"{sorted(DATASET_CONFIG)} のいずれかである必要があります。")
        return 1
    print("\n4DGS 評価サマリー (Wu et al., 2024 準拠)")
    for ds in DISPLAY_ORDER:
        if ds in data:
            print_table(ds, data[ds])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
