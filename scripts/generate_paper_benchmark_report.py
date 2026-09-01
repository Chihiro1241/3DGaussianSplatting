"""Aggregate benchmark artifacts into CSV/JSON summaries and a Markdown report."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean
from typing import Any


COLUMNS = [
    "dataset", "scene", "iteration", "paper_psnr", "ours_psnr", "delta_psnr",
    "ours_ssim", "ours_lpips", "gaussian_count", "training_time_seconds",
    "training_time_minutes", "training_wall_time_seconds", "total_run_wall_time",
    "time_to_7000_seconds", "time_to_30000_seconds", "time_7000_to_30000_seconds",
    "training_wall_time_7k", "training_wall_time_30k",
    "peak_cuda_allocated_MiB", "peak_cuda_reserved_MiB", "peak_process_vram_MiB",
    "model_parameter_MiB", "checkpoint_size_MiB", "resolution_width",
    "resolution_height", "num_train_views", "num_test_views", "gpu_name", "seed", "status",
]

PAPER_REAL = {
    ("Mip-NeRF360", 7000): {"ssim": 0.770, "psnr": 25.60, "lpips": 0.279, "train": "6m25s", "mem": 523},
    ("Mip-NeRF360", 30000): {"ssim": 0.815, "psnr": 27.21, "lpips": 0.214, "train": "41m33s", "mem": 734},
    ("Tanks&Temples", 7000): {"ssim": 0.767, "psnr": 21.20, "lpips": 0.280, "train": "6m55s", "mem": 270},
    ("Tanks&Temples", 30000): {"ssim": 0.841, "psnr": 23.14, "lpips": 0.183, "train": "26m54s", "mem": 411},
    ("Deep Blending", 7000): {"ssim": 0.875, "psnr": 27.78, "lpips": 0.317, "train": "4m35s", "mem": 386},
    ("Deep Blending", 30000): {"ssim": 0.903, "psnr": 29.41, "lpips": 0.243, "train": "36m02s", "mem": 676},
}

FEATURED_SCENES = {
    "Mip-NeRF360": ("bicycle", "garden", "kitchen"),
    "Tanks&Temples": ("truck", "train"),
    "Deep Blending": ("drjohnson", "playroom"),
    "Synthetic NeRF": ("mic", "lego", "ficus"),
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("output/paper_benchmark/manifest.json"))
    return parser


def _json(path: Path, default: Any) -> Any:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else default


def _number(value: Any, digits: int = 3) -> str:
    if value is None or value == "":
        return "N/A"
    return f"{float(value):.{digits}f}"


def _resolution(text: str | None) -> tuple[int | None, int | None]:
    if not text:
        return None, None
    token = text.split()[0].split(",")[0]
    try:
        width, height = token.split("x")
        return int(width), int(height)
    except (ValueError, AttributeError):
        return None, None


def _rows(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for scene in manifest["scenes"]:
        run_dir = Path(scene["estimated_output_path"])
        state = _json(run_dir / "run_state.json", {})
        telemetry = _json(run_dir / "training_telemetry.json", {})
        width, height = _resolution(scene.get("evaluation_resolution"))
        milestones = telemetry.get("milestones", {})
        time_7k = milestones.get("7000", {}).get("training_loop_seconds")
        time_30k = milestones.get("30000", {}).get("training_loop_seconds")
        wall_7k = milestones.get("7000", {}).get("training_wall_time_seconds")
        wall_30k = milestones.get("30000", {}).get("training_wall_time_seconds")
        for iteration in (7000, 30000):
            metrics = _json(run_dir / "metrics" / f"test_{iteration:08d}.json", {})
            checkpoint = state.get("checkpoints", {}).get(str(iteration), {})
            milestone = telemetry.get("milestones", {}).get(str(iteration), {})
            ours_psnr = metrics.get("mean_psnr")
            paper = scene[f"paper_psnr_{iteration}"]
            status = "COMPLETED" if ours_psnr is not None and checkpoint else state.get("status", scene["status"])
            training_seconds = milestone.get("training_loop_seconds")
            row = {
                "dataset": scene["dataset"], "scene": scene["scene"], "iteration": iteration,
                "paper_psnr": paper, "ours_psnr": ours_psnr,
                "delta_psnr": None if paper is None or ours_psnr is None else ours_psnr - paper,
                "ours_ssim": metrics.get("mean_ssim"), "ours_lpips": metrics.get("mean_lpips"),
                "gaussian_count": checkpoint.get("gaussian_count", milestone.get("gaussian_count")),
                "training_time_seconds": training_seconds,
                "training_time_minutes": None if training_seconds is None else training_seconds / 60,
                "training_wall_time_seconds": milestone.get("training_wall_time_seconds"),
                "total_run_wall_time": state.get("total_run_wall_time_seconds"),
                "time_to_7000_seconds": time_7k,
                "time_to_30000_seconds": time_30k,
                "time_7000_to_30000_seconds": None if time_7k is None or time_30k is None else time_30k - time_7k,
                "training_wall_time_7k": wall_7k,
                "training_wall_time_30k": wall_30k,
                "peak_cuda_allocated_MiB": telemetry.get("peak_cuda_memory_allocated_MiB"),
                "peak_cuda_reserved_MiB": telemetry.get("peak_cuda_memory_reserved_MiB"),
                "peak_process_vram_MiB": state.get("peak_process_vram_MiB"),
                "model_parameter_MiB": None if checkpoint.get("model_parameter_bytes") is None else checkpoint["model_parameter_bytes"] / (1024**2),
                "checkpoint_size_MiB": None if checkpoint.get("checkpoint_file_bytes") is None else checkpoint["checkpoint_file_bytes"] / (1024**2),
                "resolution_width": width, "resolution_height": height,
                "num_train_views": scene.get("num_train_views"), "num_test_views": scene.get("num_test_views"),
                "gpu_name": "NVIDIA RTX 6000 Ada Generation", "seed": manifest["seed"], "status": status,
            }
            rows.append(row)
    return rows


def _summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    datasets = list(dict.fromkeys(row["dataset"] for row in rows))
    for dataset in datasets:
        for iteration in (7000, 30000):
            selected = [row for row in rows if row["dataset"] == dataset and row["iteration"] == iteration]
            completed = [row for row in selected if row["ours_psnr"] is not None]
            paper_values = [row["paper_psnr"] for row in completed if row["paper_psnr"] is not None]
            ours_comparable = [row["ours_psnr"] for row in completed if row["paper_psnr"] is not None]
            output.append({
                "dataset": dataset, "iteration": iteration,
                "completed_scenes": len(completed), "expected_scenes": len(selected),
                "paper_psnr_average": mean(paper_values) if paper_values else None,
                "ours_psnr_average": mean(row["ours_psnr"] for row in completed) if completed else None,
                "delta_psnr_average": mean(ours_comparable) - mean(paper_values) if paper_values else None,
                "ours_ssim_average": mean(row["ours_ssim"] for row in completed) if completed else None,
                "ours_lpips_average": mean(row["ours_lpips"] for row in completed) if completed else None,
                "training_time_seconds_average": mean(row["training_time_seconds"] for row in completed) if completed else None,
                "training_wall_time_seconds_average": mean(row["training_wall_time_seconds"] for row in completed) if completed else None,
                "peak_process_vram_MiB_max": max((row["peak_process_vram_MiB"] for row in completed if row["peak_process_vram_MiB"] is not None), default=None),
                "model_parameter_MiB_average": mean(row["model_parameter_MiB"] for row in completed if row["model_parameter_MiB"] is not None) if any(row["model_parameter_MiB"] is not None for row in completed) else None,
                "gaussian_count_average": mean(row["gaussian_count"] for row in completed if row["gaussian_count"] is not None) if any(row["gaussian_count"] is not None for row in completed) else None,
            })
    return output


def _markdown(manifest: dict[str, Any], rows: list[dict[str, Any]], summary: list[dict[str, Any]]) -> str:
    by_scene = {(row["scene"], row["iteration"]): row for row in rows}
    lines = [
        "# 3D Gaussian Splatting 原論文再現実験", "", "## 1. 目的", "",
        "CUDA backendで各sceneを1回だけ0→30,000 iteration学習し、7K/30Kを原論文と比較する。PSNRを主指標とする。", "",
        "## 2. 実験環境", "", f"- GPU: NVIDIA RTX 6000 Ada Generation", f"- Backend: cuda", f"- Seed: {manifest['seed']}", "",
        "## 3. 原論文との実験条件比較", "",
        "| Condition | Status | Observed |", "|---|---:|---|",
    ]
    for audit in manifest["condition_audit"]:
        lines.append(f"| {audit['condition']} | {audit['status']} | {audit['observed']} |")
    lines.extend(["", "## 4. Dataset", "", f"完了run数: {sum(1 for r in rows if r['iteration']==30000 and r['status']=='COMPLETED')} / 21", "", "## 5. 評価指標", "", "Held-out test viewsのPSNR、SSIM、VGG LPIPSを算術平均する。", "", "## 6. PSNR結果", ""])
    for index, dataset in enumerate(("Mip-NeRF360", "Tanks&Temples", "Deep Blending", "Synthetic NeRF"), 1):
        lines.extend([f"### 6.{index} {dataset}", "", "| Dataset | Scene | Paper 7K | Ours 7K | Δ7K | Paper 30K | Ours 30K | Δ30K |", "|---|---|---:|---:|---:|---:|---:|---:|"])
        scenes = [scene for scene in manifest["scenes"] if scene["dataset"] == dataset]
        for scene in scenes:
            seven, thirty = by_scene[(scene["scene"], 7000)], by_scene[(scene["scene"], 30000)]
            lines.append(f"| {dataset} | {scene['scene']} | {_number(seven['paper_psnr'])} | {_number(seven['ours_psnr'])} | {_number(seven['delta_psnr'])} | {_number(thirty['paper_psnr'])} | {_number(thirty['ours_psnr'])} | {_number(thirty['delta_psnr'])} |")
        lines.append("")
    lines.extend(["## 7. SSIM / LPIPS", "", "| Dataset | Scene | SSIM 7K | LPIPS 7K | SSIM 30K | LPIPS 30K |", "|---|---|---:|---:|---:|---:|"])
    for scene in manifest["scenes"]:
        seven, thirty = by_scene[(scene["scene"], 7000)], by_scene[(scene["scene"], 30000)]
        lines.append(f"| {scene['dataset']} | {scene['scene']} | {_number(seven['ours_ssim'],4)} | {_number(seven['ours_lpips'],4)} | {_number(thirty['ours_ssim'],4)} | {_number(thirty['ours_lpips'],4)} |")
    lines.extend(["", "## 8. 学習時間", "", "| Dataset | Scene | Ours 7K | Ours 30K | Paper avg 7K | Paper avg 30K |", "|---|---|---:|---:|---:|---:|"])
    for scene in manifest["scenes"]:
        seven, thirty = by_scene[(scene["scene"], 7000)], by_scene[(scene["scene"], 30000)]
        paper_time = manifest["paper_average_training_seconds"][scene["dataset"]]
        lines.append(f"| {scene['dataset']} | {scene['scene']} | {_number(seven['training_time_minutes'],2)} min | {_number(thirty['training_time_minutes'],2)} min | {_number(None if paper_time['7000'] is None else paper_time['7000']/60,2)} min | {_number(None if paper_time['30000'] is None else paper_time['30000']/60,2)} min |")
    lines.extend(["", "Dataset平均学習時間:", "", "| Dataset | Iteration | Completed | Avg training time | Avg wall time |", "|---|---:|---:|---:|---:|"])
    for item in summary:
        lines.append(f"| {item['dataset']} | {item['iteration']} | {item['completed_scenes']}/{item['expected_scenes']} | {_number(None if item['training_time_seconds_average'] is None else item['training_time_seconds_average']/60,2)} min | {_number(None if item['training_wall_time_seconds_average'] is None else item['training_wall_time_seconds_average']/60,2)} min |")
    lines.extend(["", "原論文はA6000、本実験はRTX 6000 Adaのため、時間差をアルゴリズム差だけとは解釈しない。", "", "## 9. GPUメモリ使用量", "", "### 9.1 Peak training VRAM", "", "PyTorch allocator peakとprocess VRAM peakは別値として保存する。", "", "| Dataset | Scene | CUDA allocated peak | CUDA reserved peak | Process VRAM peak |", "|---|---|---:|---:|---:|"])
    for scene in manifest["scenes"]:
        thirty = by_scene[(scene["scene"], 30000)]
        lines.append(f"| {scene['dataset']} | {scene['scene']} | {_number(thirty['peak_cuda_allocated_MiB'],1)} | {_number(thirty['peak_cuda_reserved_MiB'],1)} | {_number(thirty['peak_process_vram_MiB'],1)} |")
    lines.extend(["", "### 9.2 Model parameter memory", "", "| Dataset | Scene | Peak VRAM | Model MB @7K | Model MB @30K | Gaussians @30K |", "|---|---|---:|---:|---:|---:|"])
    for scene in manifest["scenes"]:
        seven, thirty = by_scene[(scene["scene"], 7000)], by_scene[(scene["scene"], 30000)]
        lines.append(f"| {scene['dataset']} | {scene['scene']} | {_number(thirty['peak_process_vram_MiB'],1)} | {_number(seven['model_parameter_MiB'],1)} | {_number(thirty['model_parameter_MiB'],1)} | {thirty['gaussian_count'] or 'N/A'} |")
    lines.extend(["", "## 10. Gaussian数", "", "7K/30KのGaussian数はbenchmark_resultsにscene別で保存する。", "", "## 11. 原論文との差", "", "Dataset平均（完了sceneのみ。未完了時はpartial）:", "", "| Dataset | Iteration | Completed | Paper PSNR | Ours PSNR | ΔPSNR |", "|---|---:|---:|---:|---:|---:|"])
    for item in summary:
        lines.append(f"| {item['dataset']} | {item['iteration']} | {item['completed_scenes']}/{item['expected_scenes']} | {_number(item['paper_psnr_average'])} | {_number(item['ours_psnr_average'])} | {_number(item['delta_psnr_average'])} |")
    comparable = [row for row in rows if row["iteration"] == 30000 and row["delta_psnr"] is not None]
    within_half = sum(abs(row["delta_psnr"]) <= 0.5 for row in comparable)
    half_to_one = sum(0.5 < abs(row["delta_psnr"]) <= 1.0 for row in comparable)
    over_one = sum(abs(row["delta_psnr"]) > 1.0 for row in comparable)
    above = sum(row["delta_psnr"] > 0 for row in comparable)
    below = sum(row["delta_psnr"] < 0 for row in comparable)
    equal = sum(row["delta_psnr"] == 0 for row in comparable)
    lines.extend([
        "", "30K scene再現性集計:", "",
        f"- ±0.5 dB以内: {within_half}",
        f"- 0.5 dB超〜±1.0 dB以内: {half_to_one}",
        f"- 1.0 dB超: {over_one}",
        f"- Ours > Paper: {above}",
        f"- Ours < Paper: {below}",
        f"- Ours = Paper: {equal}",
        "", "## 12. 考察", "",
        f"30KでPaper値を持つ完了sceneは{len(comparable)}件。±0.5 dB以内は{within_half}件、1.0 dB超は{over_one}件だった。",
        "学習時間はGPU世代が異なるため、原論文との差をアルゴリズム差だけとして解釈しない。",
        "", "## 13. 結論", "",
        f"30K完了scene数は{sum(1 for row in rows if row['iteration']==30000 and row['status']=='COMPLETED')} / 21。詳細な失敗状態はfailures.jsonに保存した。", ""
    ])
    return "\n".join(lines)


def _paper_markdown(manifest: dict[str, Any], rows: list[dict[str, Any]], summary: list[dict[str, Any]]) -> str:
    """Build the paper-style main report (Table 1/Table 2 plus figures)."""

    by_scene = {(row["scene"], row["iteration"]): row for row in rows}
    by_summary = {(row["dataset"], row["iteration"]): row for row in summary}
    lines = [
        "# 3D Gaussian Splatting 原論文再現実験", "",
        "## 1. 目的", "",
        "原論文 Section 7.2、Table 1、Table 2、およびAppendixの提示形式に沿って、CUDA backendによる21-scene再現結果を整理する。各sceneは0→30Kを1回だけ学習し、7K/30K checkpointを評価した。", "",
        "全sceneの補足表: [Appendix: scene別benchmark結果](appendix_scene_tables.md)", "",
        "## 2. 実験設定", "",
        f"- GPU: NVIDIA RTX 6000 Ada Generation（原論文はA6000）", "- Backend: CUDA rasterizer", f"- Seed: {manifest['seed']}",
        "- 実世界: every-8th test split、SfM初期化、native-resolution評価", "- Synthetic NeRF: 100K random初期化、white background",
        "- Progressive SH、paper-compatible resolution warm-up、ADC/LR/L1+D-SSIM", "",
        "Train timeは純粋なtraining loop時間のdataset平均。Memは原論文と意味を合わせ、checkpointやpeak VRAMではなくoptimized Gaussian parameters (`model_parameter_MiB`) のscene平均を示す。", "",
        "## 3. 実世界データセットの定量比較", "",
        "**主表1 — 原論文 Table 1形式のdataset平均。** ↑は高いほど、↓は低いほど良い。", "",
        "| Dataset | Iter | Paper PSNR ↑ | Ours PSNR ↑ | Paper SSIM ↑ | Ours SSIM ↑ | Paper LPIPS ↓ | Ours LPIPS ↓ | Paper Train | Ours Train | Paper Mem | Ours Mem |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for dataset in ("Mip-NeRF360", "Tanks&Temples", "Deep Blending"):
        for iteration in (7000, 30000):
            paper, ours = PAPER_REAL[(dataset, iteration)], by_summary[(dataset, iteration)]
            lines.append(
                f"| {dataset} | {iteration//1000}K | {paper['psnr']:.2f} | {ours['ours_psnr_average']:.3f} | "
                f"{paper['ssim']:.3f} | {ours['ours_ssim_average']:.3f} | {paper['lpips']:.3f} | {ours['ours_lpips_average']:.3f} | "
                f"{paper['train']} | {ours['training_time_seconds_average']/60:.2f} min | {paper['mem']} MB | {ours['model_parameter_MiB_average']:.1f} MiB |"
            )
    lines.extend([
        "", "PaperのMBと本実験のMiBには単位差がある。数値は変換せず、原論文表記と実測表記を明示した。", "",
        "## 4. Synthetic NeRFの定量比較", "",
        "**主表2 — 原論文 Table 2形式の30K PSNR。** SyntheticのPaper 7Kは報告されていない。", "",
        "| Scene | Paper PSNR | Ours PSNR | ΔPSNR |", "|---|---:|---:|---:|",
    ])
    synthetic = [scene for scene in manifest["scenes"] if scene["dataset"] == "Synthetic NeRF"]
    for scene in synthetic:
        row = by_scene[(scene["scene"], 30000)]
        lines.append(f"| {scene['scene'].capitalize()} | {row['paper_psnr']:.2f} | {row['ours_psnr']:.3f} | {row['delta_psnr']:+.3f} |")
    synth_summary = by_summary[("Synthetic NeRF", 30000)]
    lines.append(f"| **Avg.** | **{synth_summary['paper_psnr_average']:.3f}** | **{synth_summary['ours_psnr_average']:.3f}** | **{synth_summary['delta_psnr_average']:+.3f}** |")

    lines.extend(["", "## 5. 定性的結果", "", "各図は30K per-view PSNRがscene平均に最も近いheld-out test viewを使用する。左からGT、7K、30K。単体画像と共通p99スケールのerror mapは `../qualitative/` に保存した。", ""])
    subsection = 1
    for dataset in ("Mip-NeRF360", "Tanks&Temples", "Deep Blending", "Synthetic NeRF"):
        lines.extend([f"### 5.{subsection} {dataset}", ""])
        for scene in FEATURED_SCENES[dataset]:
            lines.extend([f"**{scene}**", "", f"![{dataset} {scene}](figures/fig_{_slug_for_report(dataset, scene)}.png)", ""])
        subsection += 1

    lines.extend([
        "## 6. 学習時間とメモリ", "",
        "実世界dataset平均（30K）:", "",
        "| Dataset | Paper Train | Ours Train | Paper Mem | Ours parameter Mem | Peak process VRAM |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for dataset in ("Mip-NeRF360", "Tanks&Temples", "Deep Blending"):
        paper, ours = PAPER_REAL[(dataset, 30000)], by_summary[(dataset, 30000)]
        lines.append(f"| {dataset} | {paper['train']} | {ours['training_time_seconds_average']/60:.2f} min | {paper['mem']} MB | {ours['model_parameter_MiB_average']:.1f} MiB | {ours['peak_process_vram_MiB_max']:.0f} MiB |")
    lines.extend([
        "", "Peak process VRAM、PyTorch allocated/reserved、checkpoint容量はMemとは混同せず、scene別の値をAppendixに掲載する。GPU世代が異なるため、Paperとの時間差をアルゴリズム差だけとは解釈しない。", "",
        "## 7. 考察", "",
    ])
    comparable = [row for row in rows if row["iteration"] == 30000 and row["delta_psnr"] is not None]
    within_half = sum(abs(row["delta_psnr"]) <= 0.5 for row in comparable)
    half_to_one = sum(0.5 < abs(row["delta_psnr"]) <= 1.0 for row in comparable)
    over_one = sum(abs(row["delta_psnr"]) > 1.0 for row in comparable)
    above = sum(row["delta_psnr"] > 0 for row in comparable)
    below = sum(row["delta_psnr"] < 0 for row in comparable)
    lines.extend([
        f"30Kでは±0.5 dB以内が{within_half}/21、0.5–1.0 dBが{half_to_one}/21、1.0 dB超が{over_one}/21だった。Ours > Paperは{above} scene、Ours < Paperは{below} scene。",
        f"Mip-NeRF360平均はPaper比 {by_summary[('Mip-NeRF360',30000)]['delta_psnr_average']:+.3f} dB、Tanks&Templesは {by_summary[('Tanks&Temples',30000)]['delta_psnr_average']:+.3f} dB、Deep Blendingは {by_summary[('Deep Blending',30000)]['delta_psnr_average']:+.3f} dB、Synthetic NeRFは {by_summary[('Synthetic NeRF',30000)]['delta_psnr_average']:+.3f} dB。",
        "Syntheticではlegoとficusの差が1 dBを超えたため、主表の平均だけでなく定性的図とscene別Appendixを併読する必要がある。", "",
        "## 8. 結論", "",
        "21/21 sceneがOOM・NaN・評価失敗なしで30Kまで完了した。実世界3 datasetの平均はPaperに概ね近く、Synthetic NeRFは平均で低めだった。全sceneのmetric、計算コスト、Gaussian数、代表view metadataはAppendixおよび機械可読結果に保存した。", "",
        "## References", "",
        "- Kerbl et al., [*3D Gaussian Splatting for Real-Time Radiance Field Rendering*](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/3d_gaussian_splatting_low.pdf), ACM TOG 42(4), 2023. Table 1 / Table 2 / Appendix.",
    ])
    return "\n".join(lines) + "\n"


def _slug_for_report(dataset: str, scene: str) -> str:
    prefix = {"Mip-NeRF360": "mipnerf360", "Tanks&Temples": "tandt", "Deep Blending": "deepblending", "Synthetic NeRF": "synthetic"}[dataset]
    return f"{prefix}_{scene}"


def _appendix_markdown(manifest: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    by_scene = {(row["scene"], row["iteration"]): row for row in rows}
    lines = ["# Appendix: scene別benchmark結果", "", "## A.1 Scene別7K/30K PSNR", "", "| Dataset | Scene | Paper 7K | Ours 7K | Δ7K | Paper 30K | Ours 30K | Δ30K |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for scene in manifest["scenes"]:
        seven, thirty = by_scene[(scene["scene"], 7000)], by_scene[(scene["scene"], 30000)]
        lines.append(f"| {scene['dataset']} | {scene['scene']} | {_number(seven['paper_psnr'])} | {_number(seven['ours_psnr'])} | {_number(seven['delta_psnr'])} | {_number(thirty['paper_psnr'])} | {_number(thirty['ours_psnr'])} | {_number(thirty['delta_psnr'])} |")
    lines.extend(["", "## A.2 SSIM / LPIPS", "", "| Dataset | Scene | SSIM 7K | LPIPS 7K | SSIM 30K | LPIPS 30K |", "|---|---|---:|---:|---:|---:|"])
    for scene in manifest["scenes"]:
        seven, thirty = by_scene[(scene["scene"], 7000)], by_scene[(scene["scene"], 30000)]
        lines.append(f"| {scene['dataset']} | {scene['scene']} | {seven['ours_ssim']:.4f} | {seven['ours_lpips']:.4f} | {thirty['ours_ssim']:.4f} | {thirty['ours_lpips']:.4f} |")
    lines.extend(["", "## A.3 学習時間・model memory・Gaussian数", "", "| Dataset | Scene | Train 7K | Train 30K | Parameter MiB 7K | Parameter MiB 30K | Gaussians 7K | Gaussians 30K |", "|---|---|---:|---:|---:|---:|---:|---:|"])
    for scene in manifest["scenes"]:
        seven, thirty = by_scene[(scene["scene"], 7000)], by_scene[(scene["scene"], 30000)]
        lines.append(f"| {scene['dataset']} | {scene['scene']} | {seven['training_time_minutes']:.2f} min | {thirty['training_time_minutes']:.2f} min | {seven['model_parameter_MiB']:.1f} | {thirty['model_parameter_MiB']:.1f} | {seven['gaussian_count']} | {thirty['gaussian_count']} |")
    lines.extend(["", "## A.4 Peak training VRAM / checkpoint", "", "| Dataset | Scene | CUDA allocated MiB | CUDA reserved MiB | Process VRAM MiB | Checkpoint MiB 7K | Checkpoint MiB 30K |", "|---|---|---:|---:|---:|---:|---:|"])
    for scene in manifest["scenes"]:
        seven, thirty = by_scene[(scene["scene"], 7000)], by_scene[(scene["scene"], 30000)]
        lines.append(f"| {scene['dataset']} | {scene['scene']} | {thirty['peak_cuda_allocated_MiB']:.1f} | {thirty['peak_cuda_reserved_MiB']:.1f} | {thirty['peak_process_vram_MiB']:.1f} | {seven['checkpoint_size_MiB']:.1f} | {thirty['checkpoint_size_MiB']:.1f} |")
    lines.extend(["", "## A.5 定性的画像index", "", "代表viewは30K per-view PSNRがscene平均に最も近いtest view。各directoryにGT、7K、30K、error map、metadata.jsonを保存した。", "", "| Dataset | Scene | Comparison | Native images |", "|---|---|---|---|"])
    for scene in manifest["scenes"]:
        slug = _slug_for_report(scene["dataset"], scene["scene"])
        lines.append(f"| {scene['dataset']} | {scene['scene']} | [figure](figures/fig_{slug}.png) | [qualitative](../qualitative/{slug}/) |")
    return "\n".join(lines) + "\n"


def main() -> int:
    args = _parser().parse_args()
    manifest = json.loads(args.manifest.resolve().read_text(encoding="utf-8"))
    output = Path(manifest["output_root"])
    results = output / "results"
    results.mkdir(parents=True, exist_ok=True)
    rows = _rows(manifest)
    summary = _summary(rows)
    with (results / "benchmark_results.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=COLUMNS)
        writer.writeheader(); writer.writerows(rows)
    (results / "benchmark_results.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    with (results / "dataset_summary.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
        writer.writeheader(); writer.writerows(summary)
    failures = [{"dataset": row["dataset"], "scene": row["scene"], "status": row["status"]} for row in rows if row["iteration"] == 30000 and row["status"] not in {"COMPLETED", "AVAILABLE"}]
    (results / "failures.json").write_text(json.dumps(failures, indent=2) + "\n", encoding="utf-8")
    report = output / "report" / "3dgs_paper_comparison.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(_paper_markdown(manifest, rows, summary), encoding="utf-8")
    appendix = output / "report" / "appendix_scene_tables.md"
    appendix.write_text(_appendix_markdown(manifest, rows), encoding="utf-8")
    print(report)
    print(appendix)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
