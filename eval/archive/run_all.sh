#!/bin/bash
# 一括評価: eval/evaluate.py を全シーンに回し、eval/summarize.py で集計する。
#
# 前提: 各シーンの test view が ROOT_3DGS/ROOT_4DGS の <dataset>/renders/<scene>/ に
#       PNG として書き出されていること。
#       書き出しは scripts/render.py で行う:
#
#   python scripts/render.py \
#       --data data/static/nerf_synthetic/lego \
#       --checkpoint output/3DGS/nerf_synthetic/runs/lego/checkpoints/iteration_00030000.pt \
#       --split test --render-backend cuda \
#       --output output/3DGS/nerf_synthetic/renders/lego
#
#       render.py はベース名のみのフラットな PNG を出力する。evaluate.py は
#       GT 側のサブディレクトリ (nerf_synthetic の test/) をベース名で解決する。
#
# 存在しない render_dir / gt_dir のシーンは警告してスキップする
# (D-NeRF / HyperNeRF / Neu3D は現状このリポジトリに未配置)。

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# レンダリング出力のルートは output/ の分類に合わせてデータセット単位で分かれる。
# 3DGS論文のデータセット (nerf_synthetic / mipnerf360 / tandt / db) と
# 4DGS論文のデータセット (dnerf / hypernerf / neu3d) で保存先が異なる。
# 4DGS 側は output/4DGS/<dataset>/renders/<scene>/ の形。
ROOT_3DGS="${ROOT_3DGS:-$REPO_ROOT/output/3DGS}"
ROOT_4DGS="${ROOT_4DGS:-$REPO_ROOT/output/4DGS}"
# GT のルート。データセットは data/{static,dynamic}/<dataset>/ に分かれている。
# ただし neu3d のみ学習ジョブ実行中のため data/neu3d/ に据え置き（完了後に dynamic/ へ移す）。
BASE_GT="${BASE_GT:-$REPO_ROOT/data}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/eval/results}"
DEVICE="${DEVICE:-cuda}"
# RGBA の合成背景。configs/paper_benchmark/synthetic.yaml の
# data.rgba_background: white / rendering.background: [1,1,1] に対応。
SYNTHETIC_BG="${SYNTHETIC_BG:-white}"
PYTHON="${PYTHON:-python}"
mkdir -p "$OUTPUT_DIR"

skipped=0
evaluated=0

# evaluate_scene <dataset> <scene_label> <render_dir> <gt_dir> [rgba_background]
# rgba_background は RGBA の GT をどの背景へ合成するかで、学習時 config の
# data.rgba_background / rendering.background と一致させる必要がある。
# 不一致だと透明背景が黒く読まれて PSNR が 1 dB 台まで落ちる。
evaluate_scene() {
    local dataset="$1" scene="$2" render_dir="$3" gt_dir="$4" bg="${5:-white}"
    if [[ ! -d "$render_dir" ]]; then
        echo "[skip] $dataset/$scene: レンダリング出力が存在しません ($render_dir)"
        skipped=$((skipped + 1))
        return 0
    fi
    if [[ ! -d "$gt_dir" ]]; then
        echo "[skip] $dataset/$scene: GT が存在しません ($gt_dir)"
        skipped=$((skipped + 1))
        return 0
    fi
    "$PYTHON" "$REPO_ROOT/eval/evaluate.py" \
        --dataset "$dataset" \
        --render_dir "$render_dir" \
        --gt_dir "$gt_dir" \
        --output_csv "$OUTPUT_DIR/${dataset}_${scene}.csv" \
        --rgba_background "$bg" \
        --device "$DEVICE"
    evaluated=$((evaluated + 1))
}

# ---------------------------------------------------------------------------
# 本リポジトリに実在する静的データセット
# ---------------------------------------------------------------------------

# NeRF Synthetic: GT は <scene>/test/ (r_*_depth_*.png は evaluate.py 側で除外)
for scene in chair drums ficus hotdog lego materials mic ship; do
    evaluate_scene nerf_synthetic "$scene" \
        "$ROOT_3DGS/nerf_synthetic/renders/$scene" \
        "$BASE_GT/static/nerf_synthetic/$scene/test" \
        "$SYNTHETIC_BG"
done

# Mip-NeRF360: 屋外は images_4、屋内は images_2
# (scripts/run_paper_benchmark.py の _image_directory と同じ規則)
for scene in bicycle flowers garden stump treehill; do
    evaluate_scene colmap "mipnerf360_$scene" \
        "$ROOT_3DGS/mipnerf360/renders/$scene" \
        "$BASE_GT/static/mipnerf360/$scene/images_4"
done
for scene in room counter kitchen bonsai; do
    evaluate_scene colmap "mipnerf360_$scene" \
        "$ROOT_3DGS/mipnerf360/renders/$scene" \
        "$BASE_GT/static/mipnerf360/$scene/images_2"
done

# Tanks&Temples / Deep Blending: GT は <scene>/images/
for scene in train truck; do
    evaluate_scene colmap "tandt_$scene" \
        "$ROOT_3DGS/tandt/renders/$scene" \
        "$BASE_GT/static/tandt_db/tandt/$scene/images"
done
for scene in drjohnson playroom; do
    evaluate_scene colmap "deepblending_$scene" \
        "$ROOT_3DGS/deepblending/renders/$scene" \
        "$BASE_GT/static/tandt_db/db/$scene/images"
done

# ---------------------------------------------------------------------------
# 4DGS 論文 (Wu et al., 2024) のベンチマーク
# 現状このリポジトリにデータは未配置。投入すれば自動的に評価対象になる。
# ---------------------------------------------------------------------------

# 展開後の構造は data/dynamic/dnerf/<scene>/{transforms_*.json,train,test,val}/ で確認済み。
# GT の test/ は r_000.png 〜 r_019.png の 20 枚 (深度/法線ファイルの混在なし)。
for scene in bouncingballs hellwarrior hook jumpingjacks lego mutant standup trex; do
    evaluate_scene dnerf "$scene" \
        "$ROOT_4DGS/dnerf/renders/$scene" \
        "$BASE_GT/dynamic/dnerf/$scene/test" \
        "$SYNTHETIC_BG"
done

# 配布 zip の展開名はリリース資産名と一致しない。
# 実測した展開名: vrig-3dprinter / vrig-chicken / vrig-peel-banana / broom2
# (broom は "broom2"、banana は "vrig-peel-banana" として展開される)
#
# GT は配布画像 rgb/<倍率>/ ではなく eval/convert_hypernerf.py が書き出す
# converted/test/ を使う。変換器は主点オフセット・画素アスペクト比・
# フレームごとの焦点距離差を画像側のクロップで吸収しており、レンダリング画像は
# そのクロップ後の画角で出力されるため、配布画像とは画角が一致しない。
for scene in vrig-3dprinter vrig-chicken vrig-peel-banana broom2; do
    evaluate_scene hypernerf "$scene" \
        "$ROOT_4DGS/hypernerf/renders/$scene" \
        "$BASE_GT/dynamic/hypernerf/$scene/converted/test"
done

for scene in coffee_martini cook_spinach cut_beef flame_salmon flame_steak sear_steak; do
    evaluate_scene neu3d "$scene" \
        "$ROOT_4DGS/neu3d/renders/$scene" \
        "$BASE_GT/neu3d/$scene/images"
done

echo
echo "評価済み: $evaluated シーン / スキップ: $skipped シーン"

if [[ "$evaluated" -eq 0 ]]; then
    echo "[エラー] 評価できたシーンがありません。先に scripts/render.py で"
    echo "         $ROOT_3DGS または $ROOT_4DGS の <dataset>/renders/<scene>/ へ test view を書き出してください。"
    exit 1
fi

"$PYTHON" "$REPO_ROOT/eval/summarize.py" --results_dir "$OUTPUT_DIR"
