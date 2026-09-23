"""
4DGS 評価結果 集計スクリプト

eval/evaluate.py が出力した CSV を読み、データセット別にシーン一覧・平均・
論文値を並べて表示する。

``output/<3DGS|4DGS>/<dataset>/[<scene>/][<run>/]results/<name>.csv`` を再帰的にたどり、
ディレクトリ名から指標セットを決める。旧来のフラットな ``<dataset>_<scene>.csv``
も引き続き読める (eval/archive/ のスクリプトが今もこの名前で出すため)。

使い方: python eval/summarize.py --results_dir ./output
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

# ディレクトリ名 → DATASET_CONFIG のキー。実写 3 種は指標セットが同じ colmap。
DIR_ALIAS = {"mipnerf360": "colmap", "tandt": "colmap", "deepblending": "colmap"}

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
    """``<dataset>_<scene>`` を分解する (旧フラット命名の後方互換)。

    データセット名自体がアンダースコアを含む (``nerf_synthetic``) ため、
    単純な ``split("_", 1)`` では誤って ``nerf`` / ``synthetic_lego`` に割れる。
    既知のキーを長い順に照合して解決する。
    """
    for key in sorted(DATASET_CONFIG, key=len, reverse=True):
        prefix = key + "_"
        if stem.startswith(prefix) and len(stem) > len(prefix):
            return key, stem[len(prefix):]
    return None, None


def split_path(rel: Path):
    """``<3DGS|4DGS>/<dataset>/[<scene>/]results/<name>.csv`` を (dataset, scene) に分解する。

    ディレクトリ名が指標セットを決める。実写データセット 3 種は evaluate.py 側では
    まとめて ``colmap`` 扱いなので DIR_ALIAS で寄せ、どれ由来か分かるように
    ディレクトリ名を scene 側に残す。
    """
    parts = rel.with_suffix("").parts
    for i, part in enumerate(parts[:-1]):          # 最後の要素はファイル名
        dataset = DIR_ALIAS.get(part, part if part in DATASET_CONFIG else None)
        if dataset is None:
            continue
        rest = [x for x in parts[i + 1:] if x != "results"]   # 置き場の目印は scene 名ではない
        # 1 ラン 1 ディレクトリなので <run>/results/metrics.csv の "metrics" はラン名ではない
        if len(rest) > 1 and rest[-1] == "metrics":
            rest = rest[:-1]
        if dataset != part:                        # colmap は由来が消えるので補う
            rest.insert(0, part)
        return dataset, "/".join(rest)
    return None, None


def collect_results(results_dir):
    directory = Path(results_dir)
    if not directory.is_dir():
        raise FileNotFoundError(f"結果ディレクトリが存在しません: {directory}")
    data = defaultdict(dict)
    for p in sorted(directory.rglob("*.csv")):
        rel = p.relative_to(directory)
        dataset, scene = split_path(rel)
        if dataset is None:                        # 旧フラット命名へフォールバック
            dataset, scene = split_stem(p.stem)
        if dataset is None:
            continue
        # AVERAGE 行を持たない CSV (_per_frame / _gaussian_counts など) はここで落ちる
        m = parse_csv(p)
        if m:
            data[dataset][scene] = m
    return data


def print_table(dataset, scene_data):
    cfg = DATASET_CONFIG[dataset]
    metrics = cfg["metrics"]
    ref = PAPER_REFERENCE.get(dataset, {})
    # scene は階層化で "coffee_martini/baseline_30k" のように長くなりうるので
    # 列幅は中身から決める (22 は「平均 (Our Impl.)」などの見出しぶんの下限)。
    width = max(22, *(len(s) for s in scene_data))
    line = "─" * (width + 14 * len(metrics))
    print(f"\n{'═'*65}\n  {cfg['label']}  (解像度: {cfg['resolution']})\n{'═'*65}")
    print(f"  {'シーン':<{width}}" + "".join(f"  {m.upper():>10} {DIRECTIONS.get(m,'')}" for m in metrics))
    print(f"  {line}")
    scene_avgs = defaultdict(list)
    for scene, vals in sorted(scene_data.items()):
        print(f"  {scene:<{width}}" + "".join(f"  {vals.get(m, float('nan')):>12.4f}" for m in metrics))
        for m in metrics:
            scene_avgs[m].append(vals.get(m, float("nan")))
    print(f"  {line}")
    avgs = [
        float(np.nanmean(scene_avgs[m])) if np.any(np.isfinite(scene_avgs[m])) else float("nan")
        for m in metrics
    ]
    print(f"  {'平均 (Our Impl.)':<{width}}" + "".join(f"  {v:>12.4f}" for v in avgs))
    if ref:
        print(f"  {'論文値 (Wu+2024)':<{width}}" + "".join(f"  {ref.get(m, float('nan')):>12.4f}" for m in metrics))


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results_dir", default="output", help="既定は output/")
    args = p.parse_args()
    data = collect_results(args.results_dir)
    if not data:
        print(f"[エラー] 集計できる CSV が見つかりませんでした: {args.results_dir}")
        print(f"         '<3DGS|4DGS>/<dataset>/[<scene>/]results/<name>.csv' の階層か、旧来の "
              f"'<dataset>_<scene>.csv' 形式で、dataset は "
              f"{sorted(DATASET_CONFIG) + sorted(DIR_ALIAS)} のいずれかである必要があります。")
        return 1
    print("\n4DGS 評価サマリー (Wu et al., 2024 準拠)")
    for ds in DISPLAY_ORDER:
        if ds in data:
            print_table(ds, data[ds])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
