"""Render one representative held-out view per paper benchmark scene.

The representative view is the 30K test view whose PSNR is closest to the
scene mean.  Native-resolution GT, 7K/30K renders, error maps, and a compact
annotated comparison figure are written without retraining any model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from PIL import Image, ImageDraw, ImageFont

from gaussian_splatting.config import config_from_mapping, resolve_device, resolve_dtype
from gaussian_splatting.data import load_dataset
from gaussian_splatting.io.checkpoint import model_from_checkpoint_state, read_checkpoint
from gaussian_splatting.renderer import GaussianRenderer
from gaussian_splatting.training.trainer import camera_to


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("output/paper_benchmark/manifest.json"))
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _slug(dataset: str, scene: str) -> str:
    prefix = {
        "Mip-NeRF360": "mipnerf360",
        "Tanks&Temples": "tandt",
        "Deep Blending": "deepblending",
        "Synthetic NeRF": "synthetic",
    }[dataset]
    return f"{prefix}_{scene}"


def _image_directory(dataset: str, scene: str) -> str:
    if dataset == "Mip-NeRF360":
        return "images_4" if scene in {"bicycle", "flowers", "garden", "stump", "treehill"} else "images_2"
    return "images"


def _save_tensor(image: torch.Tensor, path: Path) -> None:
    pixels = image.detach().clamp(0, 1).mul(255).round().byte().permute(1, 2, 0).cpu().numpy()
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(pixels).save(path, compress_level=3)


def _heatmap(error: torch.Tensor, scale: float) -> Image.Image:
    # A compact blue -> cyan -> yellow -> red map.  The same p99 scale is used
    # for 7K and 30K within a scene, so their colors remain comparable.
    value = (error.mean(dim=0) / max(scale, 1e-8)).clamp(0, 1)
    stops = torch.tensor(
        [[0, 0, 40], [0, 70, 180], [0, 210, 210], [250, 220, 30], [220, 20, 10]],
        dtype=torch.float32,
        device=value.device,
    )
    position = value * (len(stops) - 1)
    low = position.floor().long().clamp(max=len(stops) - 2)
    frac = (position - low).unsqueeze(-1)
    rgb = stops[low] * (1 - frac) + stops[low + 1] * frac
    return Image.fromarray(rgb.round().byte().cpu().numpy())


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for path in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf"):
        if Path(path).is_file():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def _comparison(
    images: list[Image.Image], labels: list[str], title: str, destination: Path
) -> None:
    panel_width = min(640, images[0].width)
    panel_height = round(images[0].height * panel_width / images[0].width)
    resized = [im.resize((panel_width, panel_height), Image.Resampling.LANCZOS) for im in images]
    header, footer, gap = 38, 30, 6
    canvas = Image.new("RGB", (len(images) * panel_width + (len(images) - 1) * gap, header + panel_height + footer), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 8), title, fill="black", font=_font(18))
    for index, (image, label) in enumerate(zip(resized, labels, strict=True)):
        x = index * (panel_width + gap)
        canvas.paste(image, (x, header))
        box = draw.textbbox((0, 0), label, font=_font(16))
        draw.text((x + (panel_width - (box[2] - box[0])) / 2, header + panel_height + 6), label, fill="black", font=_font(16))
    destination.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(destination, compress_level=3)


def _render(state: dict[str, Any], camera: Any) -> torch.Tensor:
    config = config_from_mapping(state["config"])
    device, dtype = resolve_device(config.runtime), resolve_dtype(config.runtime)
    model = model_from_checkpoint_state(state, config, device=device, dtype=dtype)
    renderer = GaussianRenderer(config.rendering, backend="cuda")
    runtime_camera = camera_to(camera, device=device, dtype=dtype, include_image=False)
    model.eval()
    with torch.no_grad():
        image = renderer(model, runtime_camera).image.clamp(0, 1).cpu()
    del renderer, model, runtime_camera
    torch.cuda.empty_cache()
    return image


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    root = args.manifest.parent
    qualitative_root = root / "qualitative"
    figures_root = root / "report" / "figures"
    index: list[dict[str, Any]] = []

    for number, scene in enumerate(manifest["scenes"], start=1):
        dataset, name = scene["dataset"], scene["scene"]
        slug = _slug(dataset, name)
        run_dir = root / "runs" / slug
        output_dir = qualitative_root / slug
        figure = figures_root / f"fig_{slug}.png"
        metadata_path = output_dir / "metadata.json"
        required = [output_dir / f"view_000_{suffix}.png" for suffix in ("gt", "7k", "30k", "err_7k", "err_30k")]
        if not args.overwrite and metadata_path.is_file() and figure.is_file() and all(path.is_file() for path in required):
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            index.append(metadata)
            print(f"[{number}/21] {name} SKIP existing")
            continue

        metrics_30k = json.loads((run_dir / "metrics" / "test_00030000.json").read_text(encoding="utf-8"))
        metrics_7k = json.loads((run_dir / "metrics" / "test_00007000.json").read_text(encoding="utf-8"))
        mean_30k = float(metrics_30k["mean_psnr"])
        view_name, view_metrics_30k = min(
            metrics_30k["images"].items(), key=lambda item: abs(float(item[1]["psnr"]) - mean_30k)
        )
        view_metrics_7k = metrics_7k["images"][view_name]

        state_7k = read_checkpoint(run_dir / "checkpoints" / "iteration_00007000.pt", map_location="cpu")
        config = config_from_mapping(state_7k["config"])
        dataset_bundle = load_dataset(
            Path(scene["dataset_path"]), config, splits=("test",), load_points=False,
            image_directory=_image_directory(dataset, name),
        )
        candidates = [camera for camera in dataset_bundle.test if camera.image_name == view_name]
        if len(candidates) != 1 or candidates[0].image is None:
            raise RuntimeError(f"could not uniquely resolve representative test view {slug}/{view_name}")
        camera = candidates[0]
        gt = camera.image.detach().cpu().clamp(0, 1)
        render_7k = _render(state_7k, camera)
        del state_7k
        state_30k = read_checkpoint(run_dir / "checkpoints" / "iteration_00030000.pt", map_location="cpu")
        render_30k = _render(state_30k, camera)
        del state_30k, dataset_bundle

        error_7k, error_30k = (render_7k - gt).abs(), (render_30k - gt).abs()
        shared_scale = float(torch.quantile(torch.cat((error_7k.flatten(), error_30k.flatten())), 0.99).item())
        paths = {suffix: output_dir / f"view_000_{suffix}.png" for suffix in ("gt", "7k", "30k", "err_7k", "err_30k")}
        _save_tensor(gt, paths["gt"])
        _save_tensor(render_7k, paths["7k"])
        _save_tensor(render_30k, paths["30k"])
        heat_7k, heat_30k = _heatmap(error_7k, shared_scale), _heatmap(error_30k, shared_scale)
        heat_7k.save(paths["err_7k"], compress_level=3)
        heat_30k.save(paths["err_30k"], compress_level=3)
        raw_images = [Image.open(paths[key]).convert("RGB") for key in ("gt", "7k", "30k")]
        _comparison(
            raw_images,
            ["Ground Truth", f"Ours 7K ({view_metrics_7k['psnr']:.2f} dB)", f"Ours 30K ({view_metrics_30k['psnr']:.2f} dB)"],
            f"{dataset} / {name} / {view_name}", figure,
        )
        metadata = {
            "dataset": dataset, "scene": name, "slug": slug, "representative_view": view_name,
            "selection": "30K per-view PSNR closest to scene mean", "scene_mean_psnr_30k": mean_30k,
            "view_psnr_7k": float(view_metrics_7k["psnr"]), "view_psnr_30k": float(view_metrics_30k["psnr"]),
            "width": int(gt.shape[2]), "height": int(gt.shape[1]), "error_map": "mean absolute RGB error; shared 7K/30K p99 color scale",
            "error_p99_scale": shared_scale, "figure": str(figure.relative_to(root)),
        }
        output_dir.mkdir(parents=True, exist_ok=True)
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        index.append(metadata)
        print(f"[{number}/21] {name} {view_name} 7K={view_metrics_7k['psnr']:.3f} 30K={view_metrics_30k['psnr']:.3f}")

    qualitative_root.mkdir(parents=True, exist_ok=True)
    (qualitative_root / "index.json").write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
