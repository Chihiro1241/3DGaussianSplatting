"""
eval/make_eval_report.py
学習ランの成果物から eval.md (実行結果レポート) を生成する。

使い方:
    python eval/make_eval_report.py \
        --run_dir output/4DGS/neu3d/cook_spinach/cook_spinach_baseline_7k \
        --compare output/4DGS/neu3d/cook_spinach/cook_spinach_baseline_30k

runs/4DGS_baseline.sh / runs/4DGS_warmstart.sh の report 工程から自動で呼ばれる。

方針:
  * 機械的に取れる値だけを埋める。取れなかった項目は「要追記」と書き、推測しない。
  * `<!-- human -->` が付いた節 (定性的評価 / AIによる初見) は人が書く領域なので、
    再生成のときに既存 eval.md から中身をそのまま引き継ぐ。上書きしない。
  * ADC の発火回数は config から計算するのではなく train_log.jsonl で数えた実測値を出す。
    config と実測がずれていたらその事実が見えるようにするため。
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from statistics import mean, pstdev

UNKNOWN = "要追記"
HUMAN_MARK = "<!-- human -->"


# --------------------------------------------------------------- 小道具
def sh(command: list[str], cwd: Path | None = None) -> str:
    """外部コマンドの標準出力を返す。失敗したら空文字 (値は埋めない)。"""
    try:
        out = subprocess.run(
            command, cwd=cwd, capture_output=True, text=True, timeout=30
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return out.stdout.strip() if out.returncode == 0 else ""


def read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def read_jsonl(path: Path) -> list[dict]:
    try:
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    except (OSError, json.JSONDecodeError):
        return []


def fmt_int(value) -> str:
    return f"{value:,.0f}" if isinstance(value, (int, float)) else UNKNOWN


def fmt_hours(seconds: float, frames: int) -> str:
    if not seconds:
        return UNKNOWN
    per = seconds / frames if frames else 0.0
    return f"{seconds / 3600:.1f} h ({per:.0f} s/f)"


# --------------------------------------------------------------- 実行環境
def collect_environment(render_backend: str) -> dict[str, str]:
    dmi = Path("/sys/class/dmi/id")
    vendor = (dmi / "sys_vendor").read_text().strip() if (dmi / "sys_vendor").exists() else ""
    product = (dmi / "product_name").read_text().strip() if (dmi / "product_name").exists() else ""
    machine = f"{vendor} {product}".strip() or UNKNOWN

    cpu_model = cores = threads = ""
    for line in sh(["lscpu"]).splitlines():
        if line.startswith("Model name:"):
            cpu_model = line.split(":", 1)[1].strip()
        elif line.startswith("Core(s) per socket:"):
            cores = line.split(":", 1)[1].strip()
        elif line.startswith("CPU(s):") and not threads:
            threads = line.split(":", 1)[1].strip()
    cpu = f"{cpu_model} ({cores}C/{threads}T)" if cpu_model else UNKNOWN

    mem_kb = 0
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal:"):
                mem_kb = int(line.split()[1])
                break
    except OSError:
        pass
    memory = f"{mem_kb / 1024 / 1024:.0f} GiB" if mem_kb else UNKNOWN

    total, _, _ = shutil.disk_usage(".")
    storage = f"NVMe {total / 1000**4:.1f} TB"

    gpu_name = sh(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"])
    gpu = gpu_name.splitlines()[0].replace(",", " /") if gpu_name else UNKNOWN

    framework = UNKNOWN
    kernel = UNKNOWN
    sort = UNKNOWN
    try:
        import sys

        import torch

        framework = f"Python {sys.version.split()[0]} + PyTorch {torch.__version__} / CUDA {torch.version.cuda}"
        if render_backend == "cuda":
            try:
                import diff_gaussian_rasterization  # noqa: F401

                kernel = "GraphDeco 公式 `diff_gaussian_rasterization` (--render-backend cuda)"
                sort = "上記公式カーネル内部の CUB radix sort"
            except ImportError:
                kernel = "要追記 (--render-backend cuda だが diff_gaussian_rasterization を import できない)"
        else:
            kernel = f"repo 内蔵 reference バックエンド (純 PyTorch, --render-backend {render_backend})"
            sort = "torch.sort (reference バックエンド)"
    except ImportError:
        pass

    # CUB のバージョンは env の include から読む (公式カーネルがビルド時に
    # リンクした CUB と同一とは限らないのでその旨を添える)
    import os

    prefix = os.environ.get("CONDA_PREFIX", "")
    if prefix and sort != UNKNOWN and render_backend == "cuda":
        version_header = Path(prefix) / "include/cub/version.cuh"
        if version_header.exists():
            found = re.search(r"#define CUB_VERSION (\d+)", version_header.read_text())
            if found:
                raw = int(found.group(1))
                sort += (
                    f"。env の CUB は {raw // 100000}.{raw // 100 % 1000}.{raw % 100}"
                    f" (CUB_VERSION {raw})。ビルド時にリンクされた CUB と同一かは未確認"
                )

    return {
        "MACHINE": machine,
        "CPU": cpu,
        "MEMORY": memory,
        "STORAGE": storage,
        "GPU": gpu,
        "FRAMEWORK": framework,
        "KERNEL": kernel,
        "SORT": sort,
    }


# --------------------------------------------------------------- データセット
def collect_dataset(run_dir: Path, manifest: dict) -> dict[str, str]:
    frames = manifest.get("frames") or []
    source = Path(frames[0]["source"]) if frames and "source" in frames[0] else None
    data_root = source.parent if source else None          # .../converted_4d
    scene_dir = data_root.parent if data_root else None    # .../<scene>

    info: dict[str, str] = {
        "DATASET_NAME": scene_dir.name if scene_dir else UNKNOWN,
        "FRAME_COUNT": str(len(frames)) if frames else UNKNOWN,
        "SOURCE_VIDEO": UNKNOWN,
        "RESOLUTION": UNKNOWN,
        "CAMERA_COUNT": UNKNOWN,
    }
    if scene_dir and scene_dir.parent.name:
        info["DATASET_NAME"] = f"{scene_dir.parent.name} {scene_dir.name}"

    if source and source.exists():
        images = sorted((source / "images").glob("*.png"))
        info["CAMERA_COUNT"] = str(len(images)) if images else UNKNOWN
        try:
            from PIL import Image

            sizes = {Image.open(p).size for p in images[:5]}
            if sizes:
                w = sorted(s[0] for s in sizes)
                h = sorted(s[1] for s in sizes)
                info["RESOLUTION"] = (
                    f"約 {w[0]}×{h[0]}" if len(sizes) == 1
                    else f"約 {w[0]}×{h[0]}〜{w[-1]}×{h[-1]} (カメラごとに数 px 異なる)"
                )
        except ImportError:
            pass

    # 初期点群がフレーム間で共有されているか (sparse/0 がシンボリックリンクか) を見る
    info["SFM_NOTE"] = UNKNOWN
    if data_root and data_root.is_dir():
        later = sorted(data_root.glob("frame_*/sparse/0"))[1:]
        linked = [p for p in later if p.is_symlink()]
        if later and len(linked) == len(later):
            info["SFM_NOTE"] = (
                "初期点群は frame_0001 の多視点画像に COLMAP をかけた SfM 点群で、"
                "これを全フレームが共有する (`frame_NNNN/sparse/0` は frame_0001 へのシンボリックリンク)。"
            )
        elif later:
            info["SFM_NOTE"] = "初期点群はフレームごとに個別の SfM 点群である。"

    # 元動画 (cam00.mp4) の fps / 長さ / フレーム数
    if scene_dir:
        movie = scene_dir / "cam00.mp4"
        if movie.exists() and shutil.which("ffprobe"):
            probe = sh([
                "ffprobe", "-v", "error", "-select_streams", "v:0",
                "-show_entries", "stream=nb_frames,r_frame_rate,duration,width,height",
                "-of", "json", str(movie),
            ])
            stream = (read_json_text(probe) or {}).get("streams", [{}])[0]
            if stream:
                rate = stream.get("r_frame_rate", "0/1")
                try:
                    num, den = rate.split("/")
                    fps = float(num) / float(den)
                except (ValueError, ZeroDivisionError):
                    fps = 0.0
                info["SOURCE_VIDEO"] = (
                    f"{fps:.0f}fps, {float(stream.get('duration', 0)):.0f}秒, "
                    f"{stream.get('width')}×{stream.get('height')}, "
                    f"{stream.get('nb_frames')} フレームの動画データ (ffprobe 実測)"
                )
    return info


def read_json_text(text: str) -> dict:
    try:
        return json.loads(text) if text else {}
    except json.JSONDecodeError:
        return {}


# --------------------------------------------------------------- 学習条件
def collect_config(run_dir: Path) -> tuple[dict, dict[str, str]]:
    import yaml

    path = next(iter(sorted(run_dir.glob("frame_*/config.yaml"))), run_dir / "config.yaml")
    try:
        config = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        return {}, {}
    training = config.get("training", {})
    density = config.get("density_control", {})
    data = config.get("data", {})
    fields = {
        "ITERATIONS": fmt_int(training.get("iterations")),
        "SH_DEGREE": str(config.get("model", {}).get("sh_degree", UNKNOWN)),
        "BACKGROUND": str(data.get("rgba_background", UNKNOWN)),
        "TEST_EVERY": f"test_every: {data.get('test_every', UNKNOWN)}",
        "DENSIFY_SCHEDULE": (
            f"{fmt_int(density.get('densify_from_iteration'))} / "
            f"{fmt_int(density.get('densify_until_iteration'))} / "
            f"{fmt_int(density.get('densification_interval'))}"
        ),
        "OPACITY_RESET_INTERVAL": fmt_int(density.get("opacity_reset_interval")),
        "POSITION_LR": f"{training.get('position_lr_initial')} → {training.get('position_lr_final')}",
        "OTHER_LR": " / ".join(
            str(training.get(k, UNKNOWN))
            for k in ("sh_dc_lr", "sh_rest_lr", "opacity_lr", "scale_lr", "quaternion_lr")
        ),
        "LAMBDA_DSSIM": str(config.get("loss", {}).get("lambda_dssim", UNKNOWN)),
        "CONFIG_PATH": str(path),
    }
    return config, fields


def collect_adc_counts(run_dir: Path) -> str:
    """代表フレームの train_log から実際の発火回数を数える。"""
    logs = sorted(run_dir.glob("frame_*/train_log.jsonl"))
    if not logs:
        return UNKNOWN
    rows = read_jsonl(logs[len(logs) // 2])
    if not rows:
        return UNKNOWN
    events = sum(1 for r in rows if r.get("density_control_event"))
    resets = sum(1 for r in rows if r.get("opacity_reset"))
    return f"{events} 回 / {resets} 回 (frame {logs[len(logs) // 2].parent.name.split('_')[1]} の実測)"


# --------------------------------------------------------------- ラン集計
def collect_run_stats(run_dir: Path) -> dict:
    telemetry = [read_json(p) for p in sorted(run_dir.glob("frame_*/training_telemetry.json"))]
    telemetry = [t for t in telemetry if t]
    if not telemetry:
        return {}
    gauss = [t["gaussian_count"] for t in telemetry if isinstance(t.get("gaussian_count"), int)]
    secs = [t.get("training_wall_time_seconds", 0.0) for t in telemetry]
    vram = [t.get("peak_cuda_memory_reserved_MiB", 0.0) for t in telemetry]
    statuses = {t.get("status") for t in telemetry}

    logs = sorted(run_dir.glob("frame_*/train_log.jsonl"))
    initial = UNKNOWN
    if logs:
        first = read_jsonl(logs[0])
        if first:
            initial = (
                f"{first[0]['gaussian_count']:,} "
                f"(iter {first[0]['iteration']} 時点。密度制御は densify_from より後にしか発火しない)"
            )
    train_psnr = [rows[-1].get("psnr") for rows in (read_jsonl(p) for p in logs[::max(1, len(logs) // 10)]) if rows]
    train_psnr = [p for p in train_psnr if isinstance(p, (int, float))]

    return {
        "frames": len(telemetry),
        "seconds": sum(secs),
        "gauss": gauss,
        "vram": vram,
        "statuses": statuses,
        "INITIAL_GAUSSIANS": initial,
        "TRAIN_VIEW_PSNR": (
            f"平均 {mean(train_psnr):.2f} dB ({len(train_psnr)} フレーム抽出)" if train_psnr else UNKNOWN
        ),
    }


def collect_metrics(results_dir: Path, tag: str = "") -> dict:
    """results_dir の metrics/per_frame/summary を読む (tag は旧レイアウト用の接頭辞)。"""
    out: dict[str, object] = {}
    csv_path = results_dir / f"{tag}.csv" if tag else results_dir / "metrics.csv"
    if csv_path.exists():
        rows = list(csv.DictReader(csv_path.open(encoding="utf-8")))
        if rows:
            out["count"] = len(rows)
            for key in ("psnr", "d_ssim", "lpips"):
                values = [float(r[key]) for r in rows if r.get(key)]
                if values:
                    out[key] = mean(values)
    summary = read_json(results_dir / (f"{tag}_summary.json" if tag else "summary.json"))
    if summary.get("cameras"):
        out["cameras"] = {c: v["psnr"]["mean"] for c, v in sorted(summary["cameras"].items())}
    per_frame = results_dir / (f"{tag}_per_frame.csv" if tag else "per_frame.csv")
    if per_frame.exists():
        frames: dict[int, list[float]] = {}
        for r in csv.DictReader(per_frame.open(encoding="utf-8")):
            frames.setdefault(int(r["frame"]), []).append(float(r["psnr"]))
        means = {k: mean(v) for k, v in frames.items()}
        if means:
            keys = sorted(means)
            worst = min(means, key=means.get)
            best = max(means, key=means.get)
            out["trend"] = (
                f"最初の10frame {mean([means[k] for k in keys[:10]]):.2f} dB → "
                f"最後の10frame {mean([means[k] for k in keys[-10:]]):.2f} dB、"
                f"最良 frame{best}={means[best]:.2f}、最悪 frame{worst}={means[worst]:.2f}、"
                f"標準偏差 {pstdev(list(means.values())):.2f}"
            )
    return out


def mirror_results(run_dir: Path, results_root: Path) -> tuple[Path, str]:
    """run_dir から CSV の置き場を求める。

    1 ラン 1 ディレクトリなので、評価結果は常に ``<run_dir>/results/`` にある。
    ファイル名は metrics.csv / per_frame.csv / gaussian_counts.csv / summary.json 固定で、
    tag は使わない (空文字を返す)。results_root は後方互換のため残してある。
    """
    if (run_dir / "results").is_dir():
        return run_dir / "results", ""
    return results_root, run_dir.name


def result_row(label: str, run_dir: Path, results_dir: Path, tag: str) -> str:
    """評価表の 1 行を作る。取れない値は空欄ではなく「要追記」にする。"""
    stats = collect_run_stats(run_dir)
    metrics = collect_metrics(results_dir, tag)
    disk = sh(["du", "-sh", str(run_dir)]).split("\t")[0] or UNKNOWN
    if disk and disk[-1] in "KMGT":
        disk = f"{disk[:-1]} {disk[-1]}B"
    psnr = f"{metrics['psnr']:.3f}" if "psnr" in metrics else UNKNOWN
    dssim = f"{metrics['d_ssim']:.4f}" if "d_ssim" in metrics else UNKNOWN
    lpips = f"{metrics['lpips']:.4f}" if "lpips" in metrics else UNKNOWN
    if not stats:
        return f"| {label} | {psnr} | {dssim} | {lpips} | {UNKNOWN} | {UNKNOWN} | {UNKNOWN} | {disk} |"
    gauss = fmt_int(mean(stats["gauss"])) if stats["gauss"] else UNKNOWN
    vram = (
        f"平均 {mean(stats['vram']) / 1024:.1f} GiB (最大 {max(stats['vram']) / 1024:.1f})"
        if any(stats["vram"]) else UNKNOWN
    )
    return (
        f"| {label} | {psnr} | {dssim} | {lpips} | {gauss} "
        f"| {fmt_hours(stats['seconds'], stats['frames'])} | {vram} | {disk} |"
    )


# --------------------------------------------------------------- 3D ラン
def parse_stdout_command(run_dir: Path) -> dict[str, str]:
    """stdout.log に残る historical command から --data / --image-directory を拾う。"""
    log = run_dir / "stdout.log"
    if not log.exists():
        return {}
    lines = [l for l in log.read_text(encoding="utf-8", errors="replace").splitlines() if "scripts/train.py" in l]
    if not lines:
        return {}
    last = lines[-1]
    command = last.split("$ ", 1)[1] if "$ " in last else last.strip()
    data = re.search(r"--data (\S+)", command)
    image_dir = re.search(r"--image-directory (\S+)", command)
    return {
        "HISTORICAL_COMMAND": f"`{command}`",
        "DATA_PATH": data.group(1) if data else UNKNOWN,
        "IMAGE_DIRECTORY": image_dir.group(1) if image_dir else UNKNOWN,
    }


def collect_dataset_3d(run_dir: Path) -> dict[str, str]:
    state = read_json(run_dir / "run_state.json")
    info: dict[str, str] = {
        "DATASET_NAME": state.get("dataset", run_dir.parent.parent.name or UNKNOWN),
        "SCENE": state.get("scene", run_dir.name),
        "DATA_PATH": UNKNOWN,
        "IMAGE_DIRECTORY": UNKNOWN,
        "HISTORICAL_COMMAND": UNKNOWN,
        "IMAGE_SPLIT": UNKNOWN,
        "RESOLUTION": UNKNOWN,
    }
    info.update(parse_stdout_command(run_dir))

    # historical path は実行時記録なので書き換えない。現存しない場合だけ、
    # 現在の置き場所を探して併記する (experiments/README.md の運用ルール)。
    data_dir = Path(info["DATA_PATH"]) if info["DATA_PATH"] != UNKNOWN else None
    if data_dir and not data_dir.is_dir():
        # シーン名だけで探すと別データセットの同名シーン (nerf_synthetic/lego と
        # dnerf/lego など) に誤マッチするので、親ディレクトリまで一致を要求する。
        tail = f"{data_dir.parent.name}/{data_dir.name}"
        moved = sorted({p for p in Path("data").rglob(tail) if p.is_dir()})
        if len(moved) == 1:
            info["DATA_PATH"] = f"{info['DATA_PATH']} (現存せず。現在は `{moved[0]}`)"
            data_dir = moved[0]
        elif len(moved) > 1:
            found = " / ".join(f"`{m}`" for m in moved)
            info["DATA_PATH"] = f"{info['DATA_PATH']} (現存せず。候補が複数: {found})"
            data_dir = None
        else:
            info["DATA_PATH"] = f"{info['DATA_PATH']} (現存しない)"
            data_dir = None

    image_dir = None
    if data_dir and data_dir.is_dir():
        candidate = data_dir / info["IMAGE_DIRECTORY"] if info["IMAGE_DIRECTORY"] != UNKNOWN else data_dir
        if candidate.is_dir():
            image_dir = candidate
        elif (data_dir / "train").is_dir():
            # nerf_synthetic は images/ を持たず train/ test/ に分かれている
            image_dir = data_dir / "train"
    if image_dir:
        images = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg"))
        info["IMAGE_SPLIT"] = f"{len(images)} 枚 (`{image_dir}`)"
        try:
            from PIL import Image

            sizes = {Image.open(p).size for p in images[:5]}
            if len(sizes) == 1:
                w, h = sizes.pop()
                info["RESOLUTION"] = f"{w}×{h}"
            elif sizes:
                w = sorted(x for x, _ in sizes)
                h = sorted(y for _, y in sizes)
                info["RESOLUTION"] = f"{w[0]}×{h[0]}〜{w[-1]}×{h[-1]}"
        except ImportError:
            pass
    return info


def collect_run_3d(run_dir: Path, results_dir: Path, tag: str) -> dict[str, str]:
    telemetry = read_json(run_dir / "training_telemetry.json")
    state = read_json(run_dir / "run_state.json")
    rows = read_jsonl(run_dir / "train_log.jsonl")

    events = sum(1 for r in rows if r.get("density_control_event"))
    resets = sum(1 for r in rows if r.get("opacity_reset"))
    adc = f"{events} 回 / {resets} 回 (train_log.jsonl の実測)" if rows else UNKNOWN

    initial = UNKNOWN
    if rows:
        initial = (
            f"{rows[0]['gaussian_count']:,} (iter {rows[0]['iteration']} 時点。"
            "密度制御は densify_from より後にしか発火しない)"
        )

    # milestone ごとの行 (metrics JSON + telemetry + checkpoint サイズ)
    milestone_rows = []
    milestones = sorted(int(k) for k in (telemetry.get("milestones") or {}))
    for iteration in milestones:
        metrics = read_json(run_dir / f"metrics/test_{iteration:08d}.json")
        entry = (telemetry.get("milestones") or {}).get(str(iteration), {})
        ckpt = run_dir / f"checkpoints/iteration_{iteration:08d}.pt"
        cells = [
            f"{iteration:,}",
            f"{metrics['mean_psnr']:.3f}" if "mean_psnr" in metrics else UNKNOWN,
            f"{metrics['mean_ssim']:.4f}" if "mean_ssim" in metrics else UNKNOWN,
            f"{metrics['mean_lpips']:.4f}" if "mean_lpips" in metrics else UNKNOWN,
            fmt_int(entry.get("gaussian_count")),
            (
                f"{entry['training_wall_time_seconds'] / 60:.1f} 分"
                if isinstance(entry.get("training_wall_time_seconds"), (int, float))
                else UNKNOWN
            ),
            f"{ckpt.stat().st_size / 1024 ** 3:.2f} GB" if ckpt.exists() else UNKNOWN,
        ]
        milestone_rows.append("| " + " | ".join(cells) + " |")

    # 学習経過の抜粋
    progress_rows = []
    if rows:
        wanted = [r for r in rows if r["iteration"] in {1000, 7000, 15000, 30000}] or rows[:: max(1, len(rows) // 4)]
        for r in wanted:
            progress_rows.append(
                f"| {r['iteration']:,} | {r.get('loss_total', float('nan')):.4f} "
                f"| {r.get('psnr', float('nan')):.2f} | {r.get('gaussian_count', 0):,} |"
            )

    image_eval = UNKNOWN
    metrics_csv = collect_metrics(results_dir, tag)
    if "psnr" in metrics_csv:
        image_eval = (
            f"PSNR {metrics_csv['psnr']:.3f} / D-SSIM {metrics_csv.get('d_ssim', float('nan')):.4f} "
            f"/ LPIPS {metrics_csv.get('lpips', float('nan')):.4f} ({metrics_csv.get('count')} 枚、"
            f"`{results_dir / 'metrics.csv'}`)"
        )

    vram = UNKNOWN
    if telemetry.get("peak_cuda_memory_reserved_MiB"):
        vram = f"torch peak reserved {telemetry['peak_cuda_memory_reserved_MiB'] / 1024:.1f} GiB"
        attempts = [a.get("peak_process_vram_MiB") for a in state.get("attempts", [])]
        attempts = [a for a in attempts if isinstance(a, (int, float))]
        if attempts:
            vram += f" / プロセス実測 peak {max(attempts) / 1024:.1f} GiB"

    period = UNKNOWN
    if state.get("started_at") and state.get("ended_at"):
        period = f"{state['started_at']} → {state['ended_at']}"

    split = UNKNOWN
    if milestones:
        names = list(read_json(run_dir / f"metrics/test_{milestones[-1]:08d}.json").get("images", {}))
        if names:
            split = (
                f"データセット付属の test split ({len(names)} 枚。画像名が `{names[0]}`)"
                if names[0].startswith("test/")
                else f"学習画像を間引いた test split ({len(names)} 枚。画像名が `{names[0]}`)"
            )

    status = telemetry.get("status") or state.get("status") or UNKNOWN
    return {
        "ADC_EVENT_COUNTS": adc,
        "INITIAL_GAUSSIANS": initial,
        "MILESTONES": ", ".join(f"{m:,}" for m in milestones) or UNKNOWN,
        "MILESTONE_ROWS": "\n".join(milestone_rows) or f"| {UNKNOWN} | | | | | | |",
        "PROGRESS_ROWS": "\n".join(progress_rows) or f"| {UNKNOWN} | | | |",
        "IMAGE_EVAL": image_eval,
        "VRAM": vram,
        "RUN_PERIOD": period,
        "TEST_VIEW_COUNT": fmt_int(len(read_json(run_dir / f"metrics/test_{milestones[-1]:08d}.json").get("images", {}))) if milestones else UNKNOWN,
        "TEST_SPLIT": split,
        "ANOMALIES": "なし (status COMPLETED)" if status == "COMPLETED" else f"要確認: status = {status}",
        "METHOD": "静的シーン 1 本を 3DGS で学習し、milestone ごとに test view を評価する。",
    }


# --------------------------------------------------------------- git / 人の節
def collect_git(repo: Path) -> dict[str, str]:
    branch = sh(["git", "rev-parse", "--abbrev-ref", "HEAD"], repo) or UNKNOWN
    commit = sh(["git", "rev-parse", "--short", "HEAD"], repo) or UNKNOWN
    dirty = sh(["git", "status", "--porcelain"], repo)
    return {
        "GIT_BRANCH": branch,
        "GIT_COMMIT": commit,
        "GIT_DIRTY": "clean" if not dirty else f"{len(dirty.splitlines())} ファイルが未コミット",
    }


def split_sections(text: str) -> dict[str, str]:
    """'## 見出し' 単位に分解する。"""
    sections: dict[str, str] = {}
    current = None
    buffer: list[str] = []
    for line in text.splitlines():
        if line.startswith("## "):
            if current is not None:
                sections[current] = "\n".join(buffer).strip("\n")
            current = line[3:].strip()
            buffer = []
        elif current is not None:
            buffer.append(line)
    if current is not None:
        sections[current] = "\n".join(buffer).strip("\n")
    return sections


def carry_over_human_sections(rendered: str, existing: Path) -> str:
    """既存 eval.md の <!-- human --> 節を引き継ぐ (人の記述を消さない)。"""
    if not existing.exists():
        return rendered
    old = split_sections(existing.read_text(encoding="utf-8"))
    new_sections = split_sections(rendered)
    out = rendered
    for name, body in new_sections.items():
        if HUMAN_MARK not in body:
            continue
        old_body = old.get(name, "").strip()
        if not old_body or old_body.replace(HUMAN_MARK, "").strip() in ("", UNKNOWN):
            continue
        if HUMAN_MARK not in old_body:
            old_body = f"{HUMAN_MARK}\n{old_body}"
        out = out.replace(f"## {name}\n{body}", f"## {name}\n{old_body}")
    return out


def render_3d(args, run_dir: Path, tag: str, out_path: Path, template: str, results_dir: Path) -> int:
    """静的シーン 1 本のランから eval.md を書く。"""
    _, config_fields = collect_config(run_dir)
    values = {
        "TITLE": args.title or f"{tag} 実行結果",
        "GENERATED_AT": f"{datetime.now():%Y-%m-%d %H:%M:%S}",
        "VIEWER": UNKNOWN,
        "INITIALIZATION": "SfM 点群 (COLMAP)",
        "TEST_SPLIT": UNKNOWN,
        "COMMAND": f"`{args.command}`" if args.command else UNKNOWN,
        "ARTIFACTS": "\n".join(
            f"- {label}: `{path}`"
            for label, path in [
                ("学習出力", run_dir),
                ("チェックポイント評価", run_dir / "metrics"),
                ("画像ベース評価 CSV", results_dir / f"{tag}.csv"),
                ("historical command", run_dir / "stdout.log"),
            ]
            if Path(path).exists()
        ) or UNKNOWN,
    }
    values.update(collect_environment(args.render_backend))
    values.update(collect_dataset_3d(run_dir))
    values.update(config_fields)
    values.update(collect_run_3d(run_dir, results_dir, tag))
    values.update(collect_git(Path.cwd()))
    # TEST_SPLIT は collect_run_3d が metrics の画像名から実測で決める。
    # config の test_every は nerf_synthetic では使われないため当てにしない。

    rendered = re.sub(r"\{\{([A-Z_]+)\}\}", lambda m: str(values.get(m.group(1), UNKNOWN)), template)
    rendered = carry_over_human_sections(rendered, out_path)
    out_path.write_text(rendered, encoding="utf-8")
    print(f"レポートを書きました: {out_path}")
    left = rendered.count(UNKNOWN)
    if left:
        print(f"  {UNKNOWN} が {left} 箇所あります (機械的に取れなかった項目と人が書く節)")
    return 0


# --------------------------------------------------------------- main
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run_dir", type=Path, required=True)
    parser.add_argument("--tag", default=None, help="<results_dir>/<tag>.csv の tag。既定は run_dir から逆算")
    parser.add_argument("--results_dir", type=Path, default=None,
                        help="このランの CSV があるディレクトリ。既定は <run_dir>/../../results")
    parser.add_argument("--results_root", type=Path, default=Path("output"),
                        help="runs/ 配下でない run_dir を渡したときの CSV 置き場")
    parser.add_argument("--mode", choices=("auto", "3d", "4d"), default="auto",
                        help="既定 auto。run_dir に frame_* があれば 4d、無ければ 3d")
    parser.add_argument("--template", type=Path, default=None,
                        help="既定は mode に応じて eval/templates/eval_{3d,4d}.md")
    parser.add_argument("--out", type=Path, default=None, help="既定は <run_dir>/eval.md")
    parser.add_argument("--compare", type=Path, action="append", default=[],
                        help="比較表に並べる別ランの run_dir (複数可)")
    parser.add_argument("--render_backend", default="cuda")
    parser.add_argument("--command", default="", help="実行コマンド (run スクリプトから渡す)")
    parser.add_argument("--title", default=None)
    args = parser.parse_args()

    run_dir: Path = args.run_dir
    if not run_dir.is_dir():
        print(f"[中止] run_dir がありません: {run_dir}")
        return 1
    mirrored_dir, mirrored_tag = mirror_results(run_dir, args.results_root)
    tag = args.tag or mirrored_tag or run_dir.name
    results_dir = args.results_dir or mirrored_dir
    out_path = args.out or run_dir / "eval.md"
    mode = args.mode
    if mode == "auto":
        mode = "4d" if any(run_dir.glob("frame_*/")) else "3d"
    template_path = args.template or Path(f"eval/templates/eval_{mode}.md")
    template = template_path.read_text(encoding="utf-8")

    if mode == "3d":
        return render_3d(args, run_dir, tag, out_path, template, results_dir)

    manifest = read_json(run_dir / "frames_4d.json")
    stats = collect_run_stats(run_dir)
    metrics = collect_metrics(results_dir, mirrored_tag or (args.tag or ""))
    config, config_fields = collect_config(run_dir)

    frame_dirs = sorted(run_dir.glob("frame_*/training_telemetry.json"))
    period = UNKNOWN
    if frame_dirs:
        start = datetime.fromtimestamp(frame_dirs[0].stat().st_mtime)
        end = datetime.fromtimestamp(max(p.stat().st_mtime for p in frame_dirs))
        period = f"{start:%Y-%m-%d %H:%M} → {end:%Y-%m-%d %H:%M}"

    rows = [result_row(tag, run_dir, results_dir, mirrored_tag or (args.tag or ""))]
    for other in args.compare:
        if other.is_dir():
            other_dir, other_tag = mirror_results(other, args.results_root)
            rows.append(result_row(other.name, other, other_dir, other_tag))

    camera_table = UNKNOWN
    if metrics.get("cameras"):
        header = "| " + " | ".join(metrics["cameras"]) + " |"
        divider = "|" + "---|" * len(metrics["cameras"])
        values = "| " + " | ".join(f"{v:.2f}" for v in metrics["cameras"].values()) + " |"
        camera_table = "\n".join([header, divider, values])

    artifacts = "\n".join(
        f"- {label}: `{path}`"
        for label, path in [
            ("学習出力", run_dir),
            ("評価 CSV", results_dir / "metrics.csv"),
            ("フレーム別 CSV", results_dir / "per_frame.csv"),
            ("Gaussian 数推移", results_dir / "gaussian_counts.csv"),
        ]
        if Path(path).exists()
    ) or UNKNOWN

    values = {
        "TITLE": args.title or f"{tag} 実行結果",
        "GENERATED_AT": f"{datetime.now():%Y-%m-%d %H:%M:%S}",
        "VIEWER": UNKNOWN,
        "METHOD": (
            "各フレーム独立に学習する (warm-start なし)。"
            if all(f.get("initialization") == "sfm_points" for f in manifest.get("frames", [{}]))
            else "前フレームの学習結果を次フレームの初期値として引き継ぐ (warm-start)。"
        ),
        "INITIALIZATION": ", ".join(sorted({f.get("initialization", UNKNOWN) for f in manifest.get("frames", [])})) or UNKNOWN,
        "ADC_EVENT_COUNTS": collect_adc_counts(run_dir),
        "EVAL_IMAGE_COUNT": fmt_int(metrics.get("count")),
        "TEST_VIEWS": " / ".join(metrics.get("cameras", {})) or UNKNOWN,
        "RESULT_ROWS": "\n".join(rows),
        "CAMERA_TABLE": camera_table,
        "FRAME_TREND": metrics.get("trend", UNKNOWN),
        "GAUSSIAN_MEAN": fmt_int(mean(stats["gauss"])) if stats.get("gauss") else UNKNOWN,
        "GAUSSIAN_MIN": fmt_int(min(stats["gauss"])) if stats.get("gauss") else UNKNOWN,
        "GAUSSIAN_MAX": fmt_int(max(stats["gauss"])) if stats.get("gauss") else UNKNOWN,
        "GAUSSIAN_STD": fmt_int(pstdev(stats["gauss"])) if stats.get("gauss") else UNKNOWN,
        "ANOMALIES": (
            "なし (全フレーム COMPLETED)"
            if stats.get("statuses") == {"COMPLETED"}
            else f"要確認: status = {sorted(s for s in stats.get('statuses', []) if s)}"
        ),
        "ARTIFACTS": artifacts,
        "COMMAND": f"`{args.command}`" if args.command else UNKNOWN,
        "RUN_PERIOD": period,
        "INITIAL_GAUSSIANS": stats.get("INITIAL_GAUSSIANS", UNKNOWN),
        "TRAIN_VIEW_PSNR": stats.get("TRAIN_VIEW_PSNR", UNKNOWN),
    }
    values.update(collect_environment(args.render_backend))
    values.update(collect_dataset(run_dir, manifest))
    values.update(config_fields)
    values.update(collect_git(Path.cwd()))

    rendered = re.sub(r"\{\{([A-Z_]+)\}\}", lambda m: str(values.get(m.group(1), UNKNOWN)), template)
    rendered = carry_over_human_sections(rendered, out_path)
    out_path.write_text(rendered, encoding="utf-8")
    print(f"レポートを書きました: {out_path}")
    left = rendered.count(UNKNOWN)
    if left:
        print(f"  {UNKNOWN} が {left} 箇所あります (機械的に取れなかった項目と人が書く節)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
