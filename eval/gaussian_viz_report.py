"""
eval/gaussian_viz_report.py
eval/visualize_gaussians.py が書いた PNG / HTML / summary を 1 枚の
比較ページにまとめる。

使い方:
    python eval/gaussian_viz_report.py \
        --viz_dir eval/gaussian_viz \
        --frames  1 50 100 150 200 250 300

画像は同じディレクトリに在る前提で **相対パス**で参照する。絶対パスを
src に書くと、ディレクトリごと別の場所へコピーしたときに全部切れるため。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

VIEWS = (("xy", "XY (top)"), ("xz", "XZ (front)"), ("yz", "YZ (side)"))


def load_summary(viz_dir: Path, frame: int) -> dict:
    path = viz_dir / f"frame_{frame:04d}_summary.json"
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--viz_dir", type=Path, required=True)
    parser.add_argument("--frames", type=int, nargs="+", required=True)
    parser.add_argument("--title", default="Gaussian 可視化 — baseline 30,000 iter")
    parser.add_argument("--out", type=Path, default=None,
                        help="既定: <viz_dir>/comparison.html")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    destination = args.out or (args.viz_dir / "comparison.html")

    sections: list[str] = []
    counts: list[tuple[int, int]] = []

    for frame in args.frames:
        images = []
        for key, label in VIEWS:
            name = f"frame_{frame:04d}_{key}.png"
            if (args.viz_dir / name).is_file():
                images.append(
                    f'<figure><img src="{name}" alt="frame {frame} {key}" loading="lazy">'
                    f"<figcaption>{label}</figcaption></figure>"
                )
        if not images:
            continue

        summary = load_summary(args.viz_dir, frame)
        count = summary.get("gaussian_count")
        if isinstance(count, int):
            counts.append((frame, count))

        details = ""
        if summary:
            above = summary.get("opacity_above_0.5")
            details = (
                f'<div class="meta">'
                f'<span><b>{count:,}</b> Gaussians</span>'
                f'<span>iteration {summary.get("iteration", 0):,}</span>'
                f'<span>不透明度 中央値 {summary.get("opacity_median", 0):.3f}</span>'
                f'<span>不透明度 &ge; 0.5: {above:,}</span>'
                f"</div>"
            ) if isinstance(count, int) and isinstance(above, int) else ""

        html_name = f"frame_{frame:04d}_3d.html"
        link = (
            f'<a class="link" href="{html_name}">3D 散布図を開く &rarr;</a>'
            if (args.viz_dir / html_name).is_file() else ""
        )
        sections.append(
            f'<section><h2>frame {frame:04d}{link}</h2>{details}'
            f'<div class="row">{"".join(images)}</div></section>'
        )

    if not sections:
        print(f"[エラー] {args.viz_dir} に可視化画像がありません")
        return 1

    trend = ""
    if len(counts) > 1:
        rows = "".join(
            f"<tr><td>frame {frame:04d}</td><td>{count:,}</td></tr>"
            for frame, count in counts
        )
        low = min(c for _, c in counts)
        high = max(c for _, c in counts)
        trend = (
            f"<section><h2>Gaussian 数の推移</h2>"
            f"<p>最小 {low:,} / 最大 {high:,} "
            f"(振れ幅 {100 * (high - low) / max(low, 1):.1f}%)。"
            f"baseline はフレームごとに SfM 点群から独立に学習するため、"
            f"この振れがそのままフレーム間の不安定さを表す。</p>"
            f"<table>{rows}</table></section>"
        )

    page = f"""<!DOCTYPE html>
<html lang="ja"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{args.title}</title>
<style>
  :root {{ color-scheme: dark; }}
  body {{ margin:0; padding:32px; background:#131316; color:#e8e8ec;
         font-family: system-ui, -apple-system, "Hiragino Sans", "Noto Sans JP", sans-serif; }}
  h1 {{ font-size:22px; margin:0 0 8px; }}
  .lede {{ color:#a0a0aa; margin:0 0 28px; line-height:1.7; max-width:78ch; }}
  section {{ margin-bottom:40px; border-top:1px solid #2c2c33; padding-top:20px; }}
  h2 {{ font-size:17px; margin:0 0 10px; display:flex; align-items:baseline; gap:16px; }}
  .link {{ font-size:13px; font-weight:400; color:#7fb2ff; text-decoration:none; }}
  .link:hover {{ text-decoration:underline; }}
  .meta {{ display:flex; flex-wrap:wrap; gap:18px; color:#9a9aa4; font-size:13px; margin-bottom:14px; }}
  .row {{ display:flex; gap:14px; overflow-x:auto; padding-bottom:8px; }}
  figure {{ margin:0; flex:0 0 auto; }}
  figure img {{ display:block; width:360px; border-radius:6px; background:#000; }}
  figcaption {{ margin-top:6px; font-size:12px; color:#8d8d97; text-align:center; }}
  table {{ border-collapse:collapse; font-size:13px; }}
  td {{ border:1px solid #2c2c33; padding:5px 14px; }}
  td:last-child {{ text-align:right; font-variant-numeric:tabular-nums; }}
  p {{ line-height:1.7; color:#a0a0aa; max-width:78ch; }}
</style></head><body>
<h1>{args.title}</h1>
<p class="lede">各フレームのチェックポイントから Gaussian の中心座標・SH 直流成分の色・
sigmoid 後の不透明度を読み出し、上面 (XY) / 正面 (XZ) / 側面 (YZ) に投影したもの。
明るさは不透明度で重み付けした被覆度で、色は同じ重みでの平均色。
遠景の外れ値で画が潰れないよう、描画範囲は不透明度重み付き分位点で切っている。</p>
{"".join(sections)}
{trend}
</body></html>"""

    destination.write_text(page, encoding="utf-8")
    print(f"比較 HTML: {destination}")
    print(f"  収録フレーム: {len(sections)} 件")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
