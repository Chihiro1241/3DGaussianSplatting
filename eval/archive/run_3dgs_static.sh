#!/usr/bin/env bash
# eval/run_3dgs_static.sh
# 静的シーン (3DGS 論文のベンチマーク 4 データセット) を 1 シーンずつ
# 学習 -> チェックポイント評価 -> test view 描画 -> 画像ベース評価 まで通す。
#
#   eval/run_3dgs_static.sh nerf_synthetic lego
#   eval/run_3dgs_static.sh mipnerf360                 # 引数なし = 全シーン
#   eval/run_3dgs_static.sh tandt train truck
#   setsid nohup eval/run_3dgs_static.sh deepblending > logs/3dgs_deepblending.log 2>&1 &
#
# GPU は 1 枚しか無いので学習と描画は必ず直列にする。1 シーンが失敗しても
# 残りのシーンは続行し、最後に一覧と終了コードで知らせる。
#
# 環境変数での上書き:
#   RENDER_BACKEND=cuda|reference   既定 cuda
#   DEVICE=cuda|cpu                 画像ベース評価の device (既定 cuda)
#   MILESTONES="7000 30000"         保持・評価する iteration
#   DRY_RUN=1                       コマンドを表示するだけで実行しない
#
# 各シーンの設定は scripts/run_paper_benchmark.py と同一にしてある
# (config / image-directory / milestone / --disable-training-evaluation)。
# ベンチマーク一括実行との違いは、任意のシーンだけを回せることと、
# 描画と画像ベース評価 (eval/evaluate.py) まで面倒を見ることの 2 点。

set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO" || exit 1

# torch も ffmpeg も 3dgs env の中にしか無い。base の python では動かない。
source ~/miniconda3/etc/profile.d/conda.sh
conda activate 3dgs

RENDER_BACKEND="${RENDER_BACKEND:-cuda}"
DEVICE="${DEVICE:-cuda}"
MILESTONES="${MILESTONES:-7000 30000}"
DRY_RUN="${DRY_RUN:-0}"
RESULTS_DIR="$REPO/eval/results"

# 最終チェックポイントの iteration = MILESTONES の最大値。
FINAL_ITERATION=$(printf '%s\n' $MILESTONES | sort -n | tail -1)

# ---------------------------------------------------------------------------
# データセット定義
#   DATA_ROOT     : シーンディレクトリの親
#   OUT_DATASET   : output/3DGS/<OUT_DATASET>/ の名前
#   CONFIG        : 背景色が違うので synthetic と real で分かれる
#   EVAL_DATASET  : eval/evaluate.py の --dataset (指標セットの選択)
#   LABEL_PREFIX  : eval/results/<EVAL_DATASET>_<LABEL_PREFIX><scene>.csv
#                   eval/run_all.sh と同じ命名にして summarize.py に拾わせる
#   RGBA_BG       : 学習 config の data.rgba_background と必ず揃える
#                   (食い違うと透明背景が黒く読まれて PSNR が 1dB 台に落ちる)
# ---------------------------------------------------------------------------
setup_dataset () {
    case "$1" in
        nerf_synthetic)
            DATA_ROOT="data/static/nerf_synthetic"
            OUT_DATASET="nerf_synthetic"
            CONFIG="configs/paper_benchmark/synthetic.yaml"
            EVAL_DATASET="nerf_synthetic"
            LABEL_PREFIX=""
            RGBA_BG="white"
            ALL_SCENES="chair drums ficus hotdog lego materials mic ship"
            ;;
        mipnerf360)
            DATA_ROOT="data/static/mipnerf360"
            OUT_DATASET="mipnerf360"
            CONFIG="configs/paper_benchmark/real.yaml"
            EVAL_DATASET="colmap"
            LABEL_PREFIX="mipnerf360_"
            RGBA_BG="black"
            ALL_SCENES="bicycle bonsai counter flowers garden kitchen room stump treehill"
            ;;
        tandt)
            DATA_ROOT="data/static/tandt_db/tandt"
            OUT_DATASET="tandt"
            CONFIG="configs/paper_benchmark/real.yaml"
            EVAL_DATASET="colmap"
            LABEL_PREFIX="tandt_"
            RGBA_BG="black"
            ALL_SCENES="train truck"
            ;;
        deepblending)
            DATA_ROOT="data/static/tandt_db/db"
            OUT_DATASET="deepblending"
            CONFIG="configs/paper_benchmark/real.yaml"
            EVAL_DATASET="colmap"
            LABEL_PREFIX="deepblending_"
            RGBA_BG="black"
            ALL_SCENES="drjohnson playroom"
            ;;
        *)
            echo "[エラー] 未知のデータセット: $1"
            echo "使い方: $0 <nerf_synthetic|mipnerf360|tandt|deepblending> [scene ...]"
            return 1
            ;;
    esac
}

# Mip-NeRF360 だけ屋外 1/4・屋内 1/2 に縮小した画像を使う
# (scripts/run_paper_benchmark.py の _image_directory と同じ規則)。
image_directory () {
    if [ "$OUT_DATASET" != "mipnerf360" ]; then
        echo "images"; return
    fi
    case "$1" in
        bicycle|flowers|garden|stump|treehill) echo "images_4" ;;
        *)                                     echo "images_2" ;;
    esac
}

# 画像ベース評価の GT。nerf_synthetic は test/、COLMAP 系は学習に使った
# 画像ディレクトリそのもの (test split = test_every: 8 で間引いた 8 枚に 1 枚)。
ground_truth_dir () {
    local scene="$1"
    if [ "$EVAL_DATASET" = "nerf_synthetic" ]; then
        echo "$DATA_ROOT/$scene/test"
    else
        echo "$DATA_ROOT/$scene/$(image_directory "$scene")"
    fi
}

run_step () {
    echo "+ $*"
    if [ "$DRY_RUN" = "1" ]; then
        return 0
    fi
    "$@"
}

# ---------------------------------------------------------------------------
# 1 シーン分。引数: <scene>
# ---------------------------------------------------------------------------
run_scene () {
    local scene="$1"
    local data_dir="$DATA_ROOT/$scene"
    local run_dir="output/3DGS/$OUT_DATASET/runs/$scene"
    local render_dir="output/3DGS/$OUT_DATASET/renders/$scene"
    local image_dir; image_dir="$(image_directory "$scene")"
    local gt_dir; gt_dir="$(ground_truth_dir "$scene")"
    local label="${EVAL_DATASET}_${LABEL_PREFIX}${scene}"
    local final_ckpt; final_ckpt="$(printf '%s/checkpoints/iteration_%08d.pt' "$run_dir" "$FINAL_ITERATION")"

    echo ""
    echo "############################################################"
    echo "# $OUT_DATASET / $scene   開始 $(date '+%m-%d %H:%M')"
    echo "#   data   : $data_dir ($image_dir)"
    echo "#   config : $CONFIG"
    echo "#   run    : $run_dir"
    echo "############################################################"

    if [ ! -d "$data_dir" ]; then
        echo "[中止] データがありません: $data_dir"
        return 1
    fi

    # ------------------------------------------------------------ 1. 学習
    if [ -f "$final_ckpt" ]; then
        echo "--- 学習: iteration ${FINAL_ITERATION} のチェックポイントがあるのでスキップ ---"
    else
        echo "--- 学習 (${FINAL_ITERATION} iter) ---"
        local -a train_command=(
            python scripts/train.py
            --data "$data_dir" --config "$CONFIG" --output "$run_dir"
            --image-directory "$image_dir"
            --render-backend "$RENDER_BACKEND"
            --allow-existing-output
            --milestone-iterations $MILESTONES
            --disable-training-evaluation
        )
        # 途中で落ちた run は latest.pt から再開する。checkpoint_interval は
        # 1000 なので、やり直しは最大 1000 iter で済む。
        if [ -f "$run_dir/checkpoints/latest.pt" ]; then
            echo "既存の latest.pt を検出 -> 再開"
            train_command+=(--resume "$run_dir/checkpoints/latest.pt")
        fi
        run_step "${train_command[@]}" || { echo "[失敗] 学習: $scene"; return 1; }
    fi

    # -------------------------------------------- 2. チェックポイント評価
    # milestone ごとに PSNR / SSIM / LPIPS を JSON で残す (7K と 30K の比較用)。
    local iteration ckpt metrics
    for iteration in $MILESTONES; do
        ckpt="$(printf '%s/checkpoints/iteration_%08d.pt' "$run_dir" "$iteration")"
        metrics="$(printf '%s/metrics/test_%08d.json' "$run_dir" "$iteration")"
        if [ ! -f "$ckpt" ]; then
            echo "[スキップ] iteration ${iteration}: チェックポイント無し"
            continue
        fi
        if [ -f "$metrics" ]; then
            echo "[スキップ] iteration ${iteration}: 評価済み ($metrics)"
            continue
        fi
        echo ""
        echo "--- 評価 (iteration ${iteration}) ---"
        run_step python scripts/evaluate.py \
            --data "$data_dir" --checkpoint "$ckpt" --split test \
            --output "$metrics" --image-directory "$image_dir" \
            --render-backend "$RENDER_BACKEND" \
            || { echo "[失敗] 評価 iteration ${iteration}: $scene"; return 1; }
    done

    # -------------------------------------------------------- 3. test 描画
    echo ""
    echo "--- test view 描画 ---"
    run_step mkdir -p "$render_dir"
    run_step python scripts/render.py \
        --data "$data_dir" --checkpoint "$final_ckpt" --split test \
        --output "$render_dir" --image-directory "$image_dir" \
        --render-backend "$RENDER_BACKEND" \
        || { echo "[失敗] 描画: $scene"; return 1; }
    if [ "$DRY_RUN" != "1" ]; then
        echo "PNG: $(find "$render_dir" -name '*.png' | wc -l) 枚"
    fi

    # ------------------------------------------- 4. 画像ベース評価 (CSV)
    # eval/run_all.sh と同じ CSV 名にしておくと eval/summarize.py が
    # そのまま全シーン集計に使える。
    echo ""
    echo "--- 画像ベース評価 (背景 ${RGBA_BG}) ---"
    run_step mkdir -p "$RESULTS_DIR"
    run_step python eval/evaluate.py \
        --dataset "$EVAL_DATASET" \
        --render_dir "$render_dir" --gt_dir "$gt_dir" \
        --output_csv "$RESULTS_DIR/${label}.csv" \
        --rgba_background "$RGBA_BG" --device "$DEVICE" \
        || { echo "[失敗] 画像ベース評価: $scene"; return 1; }

    echo ""
    echo "### $OUT_DATASET / $scene 完了 $(date '+%m-%d %H:%M') ###"
    return 0
}

# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------
if [ "$#" -lt 1 ]; then
    echo "使い方: $0 <nerf_synthetic|mipnerf360|tandt|deepblending> [scene ...]"
    exit 2
fi

setup_dataset "$1" || exit 2
shift
SCENES="$*"
if [ -z "$SCENES" ]; then
    SCENES="$ALL_SCENES"
fi

echo "=============================================================="
echo " 3DGS 静的シーン実行  開始 $(date '+%Y-%m-%d %H:%M:%S')"
echo " データセット: $OUT_DATASET"
echo " シーン      : $SCENES"
echo " backend     : $RENDER_BACKEND / milestones: $MILESTONES"
[ "$DRY_RUN" = "1" ] && echo " DRY_RUN     : コマンドを表示するだけで実行しません"
echo "=============================================================="

succeeded=""
failed=""
for scene in $SCENES; do
    if run_scene "$scene"; then
        succeeded="$succeeded $scene"
    else
        failed="$failed $scene"
    fi
done

echo ""
echo "=============================================================="
echo " 全処理完了 $(date '+%Y-%m-%d %H:%M:%S')"
echo " 成功:${succeeded:- なし}"
echo " 失敗:${failed:- なし}"
echo "=============================================================="
if [ "$DRY_RUN" != "1" ]; then
    du -sh "output/3DGS/$OUT_DATASET" 2>/dev/null
    df -h . | tail -1
fi

[ -z "$failed" ]
