"""フレーム × iteration の2軸でガウシアンの学習過程を再生するビューワー。

``--snapshot-interval`` 付きの学習、または ``eval/extract_snapshots.py`` が
書き出した ``<run>/snapshots/*.npz`` を読み、中心座標を3D散布図として、
不透明度を色として、ガウシアン個数を折れ線として表示する。

起動::

    streamlit run eval/snapshot_viewer.py -- --run output/4DGS/neu3d/warmstart_neu3d_trial

``--run`` は省略でき、その場合はサイドバーの入力欄からパスを指定する。

再生は Plotly のネイティブ animation を使う。Streamlit を再実行させて
コマ送りする方式と違い、再生中にサーバとの往復が発生しないため滑らかに
動く。再生軸は3通り:

* ``iteration``: フレームを固定し、学習の進行を再生する
* ``frame``     : iteration を固定し、時系列方向を再生する
* ``frame x iteration``: 全グリッドをフレーム優先で連結して再生する

座標軸の範囲は全コマの和集合で固定する。Plotly の自動スケールに任せると
コマごとに視点が飛び、ガウシアンの移動が読み取れなくなるため。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from gaussian_splatting.training.snapshot import (  # noqa: E402
    combined_bounds,
    discover_snapshot_frames,
    read_snapshot,
    read_snapshot_index,
)

PLAY_AXES = ("iteration", "frame", "frame x iteration")


def parse_args() -> argparse.Namespace:
    """``streamlit run ... -- --run PATH`` の引数を読む。"""

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--run", default="")
    known, _ = parser.parse_known_args(sys.argv[1:])
    return known


# --------------------------------------------------------------------------
# データ読み込み
# --------------------------------------------------------------------------


@st.cache_data(show_spinner=False)
def load_catalogue(root: str) -> dict[int, dict]:
    """run root 配下の全フレームの index.json を読む。

    npz 本体は読まない。個数プロットと軸範囲は index のメタデータだけで
    描けるため、点群は選ばれたコマだけを遅延ロードする。
    """

    frames = discover_snapshot_frames(root)
    catalogue: dict[int, dict] = {}
    for frame, directory in frames.items():
        entries = read_snapshot_index(directory)
        catalogue[frame] = {
            "directory": str(directory),
            "entries": entries,
            "iterations": [int(entry["iteration"]) for entry in entries],
        }
    return catalogue


@st.cache_data(show_spinner=False)
def load_points(directory: str, filename: str) -> dict:
    """1コマ分の点群を読む。Streamlit のキャッシュで再生中の再読み込みを防ぐ。"""

    payload = read_snapshot(Path(directory) / filename)
    return {
        "means": payload["means"],
        "opacities": payload["opacities"],
        "num_gaussians": payload["num_gaussians"],
        "stored_points": payload["stored_points"],
    }


def build_playlist(
    catalogue: dict[int, dict],
    *,
    axis: str,
    fixed_frame: int,
    fixed_iteration: int,
) -> list[tuple[int, dict]]:
    """再生順に並んだ ``(frame, index entry)`` を返す。

    ``frame`` 軸の再生では、フレームごとに記録 iteration が揃っていない場合が
    ある（ADC のイベント記録やフレームごとの iteration 数の違い）。その場合は
    指定 iteration に最も近いコマで代替する。
    """

    if axis == "iteration":
        frame_data = catalogue[fixed_frame]
        return [(fixed_frame, entry) for entry in frame_data["entries"]]

    if axis == "frame":
        playlist: list[tuple[int, dict]] = []
        for frame in sorted(catalogue):
            entries = catalogue[frame]["entries"]
            if not entries:
                continue
            nearest = min(
                entries,
                key=lambda entry: abs(int(entry["iteration"]) - fixed_iteration),
            )
            playlist.append((frame, nearest))
        return playlist

    playlist = []
    for frame in sorted(catalogue):
        for entry in catalogue[frame]["entries"]:
            playlist.append((frame, entry))
    return playlist


def step_label(frame: int, entry: dict, *, multi_frame: bool) -> str:
    iteration = int(entry["iteration"])
    return f"f{frame:04d} / it{iteration}" if multi_frame else f"it{iteration}"


# --------------------------------------------------------------------------
# 描画
# --------------------------------------------------------------------------


def scatter_trace(
    points: dict,
    *,
    point_size: float,
    opacity_range: tuple[float, float],
    colour_by: str,
    clip: tuple[list[float], list[float]] | None = None,
) -> go.Scatter3d:
    """1コマ分の3D散布図トレースを作る。

    ``clip`` を与えた場合、その箱の外の点は描画しない。範囲外の点を残すと
    Plotly が箱の壁面に貼り付けて描き、存在しない面ができてしまう。
    """

    means = points["means"]
    opacities = points["opacities"]
    low, high = opacity_range
    keep = (opacities >= low) & (opacities <= high) & np.isfinite(means).all(axis=1)
    if clip is not None:
        lower, upper = clip
        inside = np.ones(means.shape[0], dtype=bool)
        for axis in range(3):
            inside &= (means[:, axis] >= lower[axis]) & (means[:, axis] <= upper[axis])
        keep &= inside
    means = means[keep]
    opacities = opacities[keep]

    if colour_by == "opacity":
        colour = opacities
        colourscale = "Viridis"
        colourbar_title = "α"
        cmin, cmax = 0.0, 1.0
    else:
        colour = means[:, 2] if means.size else np.zeros(0, dtype=np.float32)
        colourscale = "Turbo"
        colourbar_title = "z"
        cmin = float(colour.min()) if colour.size else 0.0
        cmax = float(colour.max()) if colour.size else 1.0

    return go.Scatter3d(
        x=means[:, 0] if means.size else [],
        y=means[:, 1] if means.size else [],
        z=means[:, 2] if means.size else [],
        mode="markers",
        marker=dict(
            size=point_size,
            color=colour,
            colorscale=colourscale,
            cmin=cmin,
            cmax=cmax,
            opacity=0.85,
            showscale=True,
            colorbar=dict(title=colourbar_title, thickness=12),
        ),
        hovertemplate="x=%{x:.3f}<br>y=%{y:.3f}<br>z=%{z:.3f}<extra></extra>",
        name="Gaussians",
    )


def build_animation(
    playlist: list[tuple[int, dict]],
    *,
    directories: dict[int, str],
    bounds: tuple[list[float], list[float]] | None,
    point_size: float,
    opacity_range: tuple[float, float],
    colour_by: str,
    frame_duration_ms: int,
    multi_frame: bool,
) -> go.Figure:
    """Play/Pause ボタンとスライダーを備えた animation figure を組み立てる。"""

    animation_frames = []
    slider_steps = []
    for index, (frame, entry) in enumerate(playlist):
        points = load_points(directories[frame], str(entry["file"]))
        label = step_label(frame, entry, multi_frame=multi_frame)
        animation_frames.append(
            go.Frame(
                name=str(index),
                data=[
                    scatter_trace(
                        points,
                        point_size=point_size,
                        opacity_range=opacity_range,
                        colour_by=colour_by,
                        clip=bounds,
                    )
                ],
                layout=go.Layout(
                    title=(
                        f"{label} — N={int(entry['num_gaussians']):,}"
                        + (
                            f" (表示 {int(entry['stored_points']):,})"
                            if int(entry["stored_points"])
                            != int(entry["num_gaussians"])
                            else ""
                        )
                    )
                ),
            )
        )
        slider_steps.append(
            dict(
                method="animate",
                label=label,
                args=[
                    [str(index)],
                    dict(
                        mode="immediate",
                        frame=dict(duration=0, redraw=True),
                        transition=dict(duration=0),
                    ),
                ],
            )
        )

    figure = go.Figure(
        data=animation_frames[0].data if animation_frames else [],
        frames=animation_frames,
    )
    if animation_frames:
        figure.update_layout(title=animation_frames[0].layout.title)

    scene = dict(aspectmode="data")
    if bounds is not None:
        lower, upper = bounds
        scene.update(
            xaxis=dict(range=[lower[0], upper[0]], title="x"),
            yaxis=dict(range=[lower[1], upper[1]], title="y"),
            zaxis=dict(range=[lower[2], upper[2]], title="z"),
            aspectmode="cube",
        )

    figure.update_layout(
        scene=scene,
        height=680,
        margin=dict(l=0, r=0, t=40, b=0),
        uirevision="keep-camera",
        sliders=[
            dict(
                active=0,
                x=0.08,
                len=0.92,
                y=0,
                pad=dict(t=40, b=10),
                currentvalue=dict(prefix="", font=dict(size=13)),
                steps=slider_steps,
            )
        ],
        updatemenus=[
            dict(
                type="buttons",
                direction="left",
                showactive=False,
                x=0,
                y=0,
                xanchor="left",
                yanchor="top",
                pad=dict(t=45, r=10),
                buttons=[
                    dict(
                        label="▶ 再生",
                        method="animate",
                        args=[
                            None,
                            dict(
                                mode="immediate",
                                fromcurrent=True,
                                frame=dict(
                                    duration=frame_duration_ms, redraw=True
                                ),
                                transition=dict(duration=0),
                            ),
                        ],
                    ),
                    dict(
                        label="⏸ 停止",
                        method="animate",
                        args=[
                            [None],
                            dict(
                                mode="immediate",
                                frame=dict(duration=0, redraw=False),
                                transition=dict(duration=0),
                            ),
                        ],
                    ),
                ],
            )
        ],
    )
    return figure


def build_count_figure(
    catalogue: dict[int, dict], *, highlight: tuple[int, int] | None
) -> go.Figure:
    """フレームごとのガウシアン個数を iteration に対して描く。"""

    figure = go.Figure()
    for frame in sorted(catalogue):
        entries = catalogue[frame]["entries"]
        figure.add_trace(
            go.Scatter(
                x=[int(entry["iteration"]) for entry in entries],
                y=[int(entry["num_gaussians"]) for entry in entries],
                mode="lines+markers",
                name=f"frame {frame:04d}",
                marker=dict(size=4),
            )
        )
    if highlight is not None:
        figure.add_vline(
            x=highlight[1], line_width=1, line_dash="dot", line_color="#888"
        )
    figure.update_layout(
        height=300,
        margin=dict(l=0, r=0, t=30, b=0),
        title="ガウシアン個数の推移",
        xaxis_title="iteration",
        yaxis_title="ガウシアン個数",
        legend=dict(orientation="h", y=-0.25),
    )
    return figure


# --------------------------------------------------------------------------
# アプリ本体
# --------------------------------------------------------------------------


def main() -> None:
    st.set_page_config(page_title="3DGS 学習過程ビューワー", layout="wide")
    st.title("3DGS 学習過程ビューワー")

    args = parse_args()
    with st.sidebar:
        st.header("データ")
        root = st.text_input(
            "run root",
            value=args.run,
            help="単一シーンの run ディレクトリ、または frame_*/ を含む 4D run root",
        )

    if not root:
        st.info(
            "サイドバーに run のパスを入力してください。\n\n"
            "スナップショットが未生成なら、学習時に `--snapshot-interval 500` を付けるか、"
            "既存 run に対して `python eval/extract_snapshots.py --run <run>` を実行します。"
        )
        return

    try:
        catalogue = load_catalogue(root)
    except (FileNotFoundError, NotADirectoryError) as error:
        st.error(str(error))
        return

    frames = sorted(catalogue)
    all_iterations = sorted(
        {iteration for frame in frames for iteration in catalogue[frame]["iterations"]}
    )
    if not all_iterations:
        st.error("スナップショットが1件もありません。")
        return
    multi_frame = len(frames) > 1

    with st.sidebar:
        st.caption(
            f"{len(frames)} フレーム / "
            f"{sum(len(catalogue[f]['entries']) for f in frames)} スナップショット"
        )

        st.header("再生")
        axis_options = list(PLAY_AXES) if multi_frame else ["iteration"]
        axis = st.radio("再生軸", axis_options, index=0)

        fixed_frame = frames[0]
        fixed_iteration = all_iterations[-1]
        if axis == "iteration" and multi_frame:
            fixed_frame = st.select_slider("固定するフレーム", options=frames, value=frames[0])
        if axis == "frame":
            fixed_iteration = st.select_slider(
                "固定する iteration",
                options=all_iterations,
                value=all_iterations[-1],
                help="そのフレームに同じ iteration がなければ最も近いコマを使う",
            )

        fps = st.slider("再生速度 (fps)", min_value=1, max_value=30, value=8)
        max_steps = st.slider(
            "読み込む最大コマ数",
            min_value=10,
            max_value=400,
            value=120,
            step=10,
            help="超える場合は等間隔に間引く。ブラウザに送るデータ量の上限。",
        )

        st.header("表示")
        colour_by = st.radio("色", ("opacity", "depth (z)"), index=0)
        point_size = st.slider("点サイズ", 0.5, 6.0, 1.6, step=0.1)
        opacity_range = st.slider(
            "不透明度フィルタ",
            0.0,
            1.0,
            (0.0, 1.0),
            step=0.01,
            help="この範囲外の不透明度を持つガウシアンを非表示にする",
        )
        lock_axes = st.checkbox(
            "座標軸を全コマで固定",
            value=True,
            help="解除すると各コマで自動スケールし、再生中に視点が飛ぶ",
        )
        trim_outliers = st.checkbox(
            "外れ値を除いて軸を決める",
            value=True,
            disabled=not lock_axes,
            help="最適化は本体から遠く離れたガウシアンを少数残す。"
            "真の min/max に合わせるとシーン本体が点に潰れるため、"
            "既定では各軸の中央99%を使う。",
        )

    playlist = build_playlist(
        catalogue,
        axis=axis,
        fixed_frame=fixed_frame,
        fixed_iteration=fixed_iteration,
    )
    if not playlist:
        st.error("選択した条件に該当するスナップショットがありません。")
        return

    if len(playlist) > max_steps:
        picks = np.unique(
            np.linspace(0, len(playlist) - 1, num=max_steps).round().astype(int)
        )
        playlist = [playlist[index] for index in picks]

    bounds = (
        combined_bounds(
            [entry for _, entry in playlist],
            key="robust_bounds" if trim_outliers else "bounds",
        )
        if lock_axes
        else None
    )
    if lock_axes and bounds is None:
        st.warning(
            "スナップショットに座標範囲が記録されていないため軸を固定できません。"
            "`eval/extract_snapshots.py --run <run> --overwrite` で作り直してください。"
        )

    directories = {frame: catalogue[frame]["directory"] for frame in frames}
    with st.spinner(f"{len(playlist)} コマを読み込み中…"):
        figure = build_animation(
            playlist,
            directories=directories,
            bounds=bounds,
            point_size=point_size,
            opacity_range=opacity_range,
            colour_by="opacity" if colour_by == "opacity" else "depth",
            frame_duration_ms=int(1000 / fps),
            multi_frame=multi_frame,
        )

    st.plotly_chart(figure, width="stretch")
    st.caption(
        f"再生軸: {axis} / {len(playlist)} コマ — "
        "▶ 再生でアニメーション、スライダーで任意のコマへ。"
        "視点はドラッグで回転でき、再生中も維持される。"
    )

    highlight = (playlist[0][0], int(playlist[0][1]["iteration"]))
    st.plotly_chart(
        build_count_figure(catalogue, highlight=highlight if axis == "frame" else None),
        width="stretch",
    )


# ``streamlit run`` はスクリプトを __main__ として実行するため、この分岐で足りる。
if __name__ == "__main__":
    main()
