"""
eval/visualize_gaussians.py
チェックポイントから Gaussian の位置・色・不透明度を読み込み、
上面/正面/側面の 2D 投影 (PNG) と 3D 散布図 (HTML) を生成する。

使い方:
    python eval/visualize_gaussians.py \
        --ckpt_dir output/4DGS/neu3d/baseline_30k/frame_0001 \
        --out_dir  eval/gaussian_viz \
        --frame    1

----------------------------------------------------------------------------
なぜ matplotlib を使わないのか
----------------------------------------------------------------------------
30,000 iter の 1 フレームは数十万〜100 万個の Gaussian を持つ。matplotlib の
scatter はこの規模で極端に遅く、点が重なって潰れるだけで密度が読めない。
また学習が数日走っている最中に環境へ新しい依存を入れたくない
(numpy が上がると torch 側が巻き込まれる)。

そこで numpy でフレームバッファへ直接アキュムレートする。各 Gaussian を
不透明度 α の重みとして

    color_sum += rgb * α ,  alpha_sum += α

に足し込み、最後に color_sum / alpha_sum で色を、alpha_sum で被覆度を出す。
この足し込みは順序に依存しないので、奥行きソートをしなくても結果が安定する
(通常の over 合成は描画順で絵が変わってしまう)。

----------------------------------------------------------------------------
色と不透明度の復元
----------------------------------------------------------------------------
チェックポイントは生パラメータを持つので、保存形式に合わせて戻す:

    RGB     = 0.5 + C0 * sh_dc          (C0 = 0.28209479177387814, SH の直流成分)
    opacity = sigmoid(raw_opacities)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

# gaussian_splatting/math/spherical_harmonics.py と同じ 0 次の基底係数。
SH_C0 = 0.28209479177387814

VIEWS = (
    # (名前, 横軸 index, 縦軸 index, 横軸ラベル, 縦軸ラベル, 説明)
    ("xy", 0, 1, "X", "Y", "top"),
    ("xz", 0, 2, "X", "Z", "front"),
    ("yz", 1, 2, "Y", "Z", "side"),
)


def find_checkpoint(ckpt_dir: Path) -> Path:
    """フレーム出力ディレクトリから最終チェックポイントを選ぶ。

    ``latest.pt`` は最後に保存した numbered checkpoint へのハードリンクなので
    通常はこれで足りる。無い場合だけ番号順の最後にフォールバックする。
    """
    if ckpt_dir.is_file():
        return ckpt_dir
    candidates = ckpt_dir / "checkpoints"
    if not candidates.is_dir():
        candidates = ckpt_dir
    latest = candidates / "latest.pt"
    if latest.is_file():
        return latest
    numbered = sorted(candidates.glob("iteration_*.pt"))
    if not numbered:
        raise FileNotFoundError(f"チェックポイントが見つかりません: {ckpt_dir}")
    return numbered[-1]


def load_gaussians(checkpoint: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """位置 (N,3) / 色 (N,3) in [0,1] / 不透明度 (N,) in [0,1] / iteration を返す。"""
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = state["model_state_dict"]

    xyz = model["means_world"].float().numpy()
    # sh_dc は (N, 1, 3)。直流成分だけで view-independent な基準色になる。
    sh_dc = model["sh_dc"].float().numpy().reshape(len(xyz), 3)
    rgb = np.clip(0.5 + SH_C0 * sh_dc, 0.0, 1.0)
    opacity = torch.sigmoid(model["raw_opacities"].float()).numpy().reshape(-1)

    return xyz, rgb, opacity, int(state.get("iteration", -1))


def robust_bounds(
    values: np.ndarray, weights: np.ndarray, quantile: float
) -> tuple[float, float]:
    """外れ値の Gaussian で画がズームアウトしないよう分位点で範囲を決める。

    Neu3D は屋内シーンの奥に遠景の Gaussian が長く尾を引く (X の 99% 点は
    31 なのに 99.9% 点は 51、Z は 86 まで伸びる)。素の min/max で枠を取ると
    主要被写体が隅に潰れて構造が読めないので、不透明度で重み付けした
    ``quantile`` / ``1 - quantile`` 点で切る。
    """
    order = np.argsort(values)
    sorted_values = values[order]
    cumulative = np.cumsum(weights[order])
    total = cumulative[-1]
    if total <= 0:
        return float(sorted_values[0]), float(sorted_values[-1])
    low = float(np.interp(quantile * total, cumulative, sorted_values))
    high = float(np.interp((1.0 - quantile) * total, cumulative, sorted_values))
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        return float(sorted_values[0]), float(sorted_values[-1]) + 1e-6
    return low, high


def accumulate(
    horizontal: np.ndarray,
    vertical: np.ndarray,
    rgb: np.ndarray,
    opacity: np.ndarray,
    size: int,
    bounds: tuple[float, float, float, float],
    radius: int,
) -> np.ndarray:
    """不透明度を重みにした順序非依存の色アキュムレーション。"""
    left, right, bottom, top = bounds
    # 縦軸は画像座標では上下反転する (数学の +Y を上に描く)。
    column = (horizontal - left) / (right - left) * (size - 1)
    row = (top - vertical) / (top - bottom) * (size - 1)

    inside = (
        np.isfinite(column) & np.isfinite(row)
        & (column >= 0) & (column <= size - 1)
        & (row >= 0) & (row <= size - 1)
    )
    column = column[inside].astype(np.int32)
    row = row[inside].astype(np.int32)
    weight = opacity[inside].astype(np.float64)
    colors = rgb[inside].astype(np.float64)

    cells = size * size
    color_flat = np.zeros((cells, 3), dtype=np.float64)
    alpha_flat = np.zeros(cells, dtype=np.float64)

    # 同じ画素へ何万個も落ちるので、散布加算 (bincount) で畳み込む。
    # np.add.at でも書けるが桁違いに遅く、しかも reshape のビューへ
    # 書き戻す形になって意図が読みにくい。
    premultiplied = colors * weight[:, None]

    # radius>0 のときは小さな正方形に広げる。1 画素だと点が細かすぎて
    # 構造 (人物・テーブル・背景の層) が見えないため。
    offsets = range(-radius, radius + 1)
    for dr in offsets:
        rows = np.clip(row + dr, 0, size - 1).astype(np.int64)
        for dc in offsets:
            cols = np.clip(column + dc, 0, size - 1).astype(np.int64)
            flat = rows * size + cols
            for channel in range(3):
                color_flat[:, channel] += np.bincount(
                    flat, weights=premultiplied[:, channel], minlength=cells
                )
            alpha_flat += np.bincount(flat, weights=weight, minlength=cells)

    color_sum = color_flat.reshape(size, size, 3)
    alpha_sum = alpha_flat.reshape(size, size)
    return np.concatenate([color_sum, alpha_sum[..., None]], axis=-1)


def render_projection(
    accumulated: np.ndarray, gamma: float
) -> np.ndarray:
    """アキュムレート結果を黒背景の RGB 画像にする。"""
    color_sum = accumulated[..., :3]
    alpha_sum = accumulated[..., 3]

    with np.errstate(invalid="ignore", divide="ignore"):
        mean_color = np.where(
            alpha_sum[..., None] > 0, color_sum / np.maximum(alpha_sum[..., None], 1e-12), 0.0
        )

    # 被覆度は数百まで振れるので、見やすい範囲へ圧縮してから gamma をかける。
    positive = alpha_sum[alpha_sum > 0]
    scale = np.percentile(positive, 99.0) if positive.size else 1.0
    coverage = np.clip(alpha_sum / max(scale, 1e-12), 0.0, 1.0) ** gamma

    image = np.clip(mean_color * coverage[..., None], 0.0, 1.0)
    return (image * 255.0 + 0.5).astype(np.uint8)


def annotate(
    image: np.ndarray,
    *,
    view_name: str,
    description: str,
    horizontal_label: str,
    vertical_label: str,
    bounds: tuple[float, float, float, float],
    frame: int,
    count: int,
    iteration: int,
    margin: int,
) -> Image.Image:
    """軸ラベル・目盛り値・Gaussian 数を余白に描く。"""
    size = image.shape[0]
    canvas = Image.new("RGB", (size + 2 * margin, size + 2 * margin), (18, 18, 20))
    canvas.paste(Image.fromarray(image), (margin, margin))
    draw = ImageDraw.Draw(canvas)

    frame_box = (margin - 1, margin - 1, margin + size, margin + size)
    draw.rectangle(frame_box, outline=(90, 90, 96))

    left, right, bottom, top = bounds
    white = (235, 235, 240)
    grey = (150, 150, 158)

    draw.text((margin, 10), f"frame {frame:04d} / {view_name.upper()} {description}", fill=white)
    draw.text(
        (margin, 24),
        f"{count:,} Gaussians  (iteration {iteration:,})",
        fill=grey,
    )

    # 横軸: 左右端の値と軸名。
    baseline = margin + size + 6
    draw.text((margin, baseline), f"{left:.2f}", fill=grey)
    draw.text((margin + size - 34, baseline), f"{right:.2f}", fill=grey)
    draw.text((margin + size // 2 - 4, baseline + 14), horizontal_label, fill=white)

    # 縦軸: 上下端の値と軸名。
    draw.text((6, margin - 2), f"{top:.2f}", fill=grey)
    draw.text((6, margin + size - 12), f"{bottom:.2f}", fill=grey)
    draw.text((6, margin + size // 2), vertical_label, fill=white)

    return canvas


def write_projections(
    xyz: np.ndarray,
    rgb: np.ndarray,
    opacity: np.ndarray,
    *,
    out_dir: Path,
    frame: int,
    iteration: int,
    size: int,
    radius: int,
    gamma: float,
    quantile: float,
) -> list[Path]:
    written: list[Path] = []
    for name, horizontal_axis, vertical_axis, horizontal_label, vertical_label, description in VIEWS:
        horizontal = xyz[:, horizontal_axis]
        vertical = xyz[:, vertical_axis]
        left, right = robust_bounds(horizontal, opacity, quantile)
        bottom, top = robust_bounds(vertical, opacity, quantile)

        # 投影のアスペクト比を歪めないよう、広いほうの幅に正方形を合わせる。
        span = max(right - left, top - bottom)
        horizontal_centre = 0.5 * (left + right)
        vertical_centre = 0.5 * (bottom + top)
        bounds = (
            horizontal_centre - span / 2,
            horizontal_centre + span / 2,
            vertical_centre - span / 2,
            vertical_centre + span / 2,
        )

        accumulated = accumulate(
            horizontal, vertical, rgb, opacity, size, bounds, radius
        )
        image = render_projection(accumulated, gamma)
        canvas = annotate(
            image,
            view_name=name,
            description=description,
            horizontal_label=horizontal_label,
            vertical_label=vertical_label,
            bounds=bounds,
            frame=frame,
            count=len(xyz),
            iteration=iteration,
            margin=48,
        )
        path = out_dir / f"frame_{frame:04d}_{name}.png"
        canvas.save(path)
        written.append(path)
    return written


def write_html(
    xyz: np.ndarray,
    rgb: np.ndarray,
    opacity: np.ndarray,
    *,
    out_dir: Path,
    frame: int,
    iteration: int,
    thresholds: list[float],
    max_points: int,
    seed: int,
) -> Path:
    """不透明度スライダー付きの 3D 散布図 HTML。

    しきい値ごとに 1 トレースを作り、スライダーで表示を切り替える。plotly の
    スライダーはデータの再フィルタができず restyle しかできないため。
    全点を入れるとファイルが数百 MB になるので、各トレースは max_points まで
    ランダムに間引く (間引き率は凡例に出す)。
    """
    import plotly.graph_objects as go

    generator = np.random.default_rng(seed)
    figure = go.Figure()

    for index, threshold in enumerate(thresholds):
        selected = np.flatnonzero(opacity >= threshold)
        total = selected.size
        if total > max_points:
            selected = generator.choice(selected, size=max_points, replace=False)
        shown = selected.size

        colours = [
            f"rgb({int(r * 255)},{int(g * 255)},{int(b * 255)})"
            for r, g, b in rgb[selected]
        ]
        figure.add_trace(
            go.Scatter3d(
                x=xyz[selected, 0],
                y=xyz[selected, 1],
                z=xyz[selected, 2],
                mode="markers",
                marker=dict(size=1.2, color=colours, opacity=0.8),
                name=f"opacity >= {threshold:.2f}",
                visible=(index == 0),
                hoverinfo="skip",
                showlegend=False,
            )
        )
        figure.data[index].customdata = None
        figure.data[index].meta = dict(total=int(total), shown=int(shown))

    steps = []
    for index, threshold in enumerate(thresholds):
        meta = figure.data[index].meta
        steps.append(
            dict(
                method="update",
                label=f"{threshold:.2f}",
                args=[
                    {"visible": [i == index for i in range(len(thresholds))]},
                    {
                        "title": (
                            f"frame {frame:04d} — iteration {iteration:,} — "
                            f"opacity >= {threshold:.2f} : "
                            f"{meta['total']:,} Gaussians "
                            f"(表示 {meta['shown']:,})"
                        )
                    },
                ],
            )
        )

    first = figure.data[0].meta
    figure.update_layout(
        title=(
            f"frame {frame:04d} — iteration {iteration:,} — "
            f"opacity >= {thresholds[0]:.2f} : {first['total']:,} Gaussians "
            f"(表示 {first['shown']:,})"
        ),
        sliders=[dict(active=0, currentvalue={"prefix": "不透明度しきい値: "}, steps=steps)],
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
            aspectmode="data",
            bgcolor="rgb(12,12,14)",
        ),
        paper_bgcolor="rgb(24,24,27)",
        font=dict(color="rgb(230,230,235)"),
        margin=dict(l=0, r=0, t=48, b=0),
        height=780,
    )

    path = out_dir / f"frame_{frame:04d}_3d.html"
    # "directory" は plotly.min.js を出力先に 1 つだけ置いて相対参照する。
    # "cdn" だとオフラインの機械で白紙になり、True だと 7 フレームぶん
    # ライブラリを重複して抱えることになる。
    figure.write_html(path, include_plotlyjs="directory")
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--ckpt_dir", type=Path, required=True,
                        help="frame_NNNN/ (中の checkpoints/latest.pt を読む) "
                             "か、チェックポイントファイルそのもの")
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--frame", type=int, required=True)
    parser.add_argument("--size", type=int, default=900,
                        help="2D 投影 1 枚の描画領域の 1 辺 (px)")
    parser.add_argument("--radius", type=int, default=1,
                        help="1 Gaussian を広げる半径 (px)。0 で 1 画素")
    parser.add_argument("--quantile", type=float, default=0.03,
                        help="描画範囲を決める不透明度重み付き分位点。"
                             "小さいほど遠景まで入る")
    parser.add_argument("--gamma", type=float, default=0.45,
                        help="被覆度のトーンカーブ。小さいほど疎な部分が見える")
    parser.add_argument("--max_points", type=int, default=20000,
                        help="HTML の 1 トレースに入れる最大点数")
    parser.add_argument("--thresholds", type=float, nargs="+",
                        default=[0.0, 0.05, 0.1, 0.25, 0.5, 0.75],
                        help="HTML スライダーの不透明度しきい値")
    parser.add_argument("--skip_html", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = find_checkpoint(args.ckpt_dir)
    xyz, rgb, opacity, iteration = load_gaussians(checkpoint)
    print(f"[frame {args.frame}] {checkpoint}: {len(xyz):,} Gaussians "
          f"(iteration {iteration:,})")

    written = write_projections(
        xyz, rgb, opacity,
        out_dir=args.out_dir, frame=args.frame, iteration=iteration,
        size=args.size, radius=args.radius, gamma=args.gamma,
        quantile=args.quantile,
    )
    for path in written:
        print(f"  2D: {path}")

    if not args.skip_html:
        html = write_html(
            xyz, rgb, opacity,
            out_dir=args.out_dir, frame=args.frame, iteration=iteration,
            thresholds=list(args.thresholds), max_points=args.max_points,
            seed=args.seed,
        )
        print(f"  3D: {html}")

    summary = {
        "frame": args.frame,
        "checkpoint": str(checkpoint),
        "iteration": iteration,
        "gaussian_count": int(len(xyz)),
        "opacity_mean": float(opacity.mean()),
        "opacity_median": float(np.median(opacity)),
        "opacity_above_0.5": int((opacity >= 0.5).sum()),
        "extent_min": [float(v) for v in xyz.min(axis=0)],
        "extent_max": [float(v) for v in xyz.max(axis=0)],
    }
    summary_path = args.out_dir / f"frame_{args.frame:04d}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"  summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
