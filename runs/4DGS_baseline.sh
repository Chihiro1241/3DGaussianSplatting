#!/usr/bin/env bash
# 4DGS_baseline.sh — Neu3D の動的シーンを「フレームごとに独立学習」で回す。
# フレーム間の受け渡しを一切しない baseline。各フレームがその frame の
# SfM 点群から学習を始めるので、warm-start の比較対象になる。
#
#   runs/4DGS_baseline.sh                          # coffee_martini / 300 フレーム
#   runs/4DGS_baseline.sh cook_spinach 100
#   setsid nohup runs/4DGS_baseline.sh > logs/4dgs_baseline.log 2>&1 &
#
# tmux / screen は入っていないので、長時間ジョブは setsid nohup で detach する。
#
# 工程: 学習 -> マニフェスト再構築 -> test view 描画 -> 評価 -> 動画 ->
#       Gaussian 可視化 -> eval.md 生成。GPU は 1 枚なので学習と描画は必ず直列にする。
#
# 環境変数:
#   CONFIG="configs/neu3d/baseline_7k.yaml"  学習設定
#   TAG="<scene>_<config 名>"                出力ディレクトリの名前
#   RUN_DIR="output/4DGS/neu3d/<SCENE>/<TAG>"  ラン一式 (学習/描画/動画/可視化/評価)
#   STAGES="train render eval video viz report"  実行する工程 (既定は全部)
#   COMPARE="<run_dir> ..."                  eval.md の比較表に並べる別ラン
#   VIZ_FRAMES="1 50 100 ..."                Gaussian 可視化するフレーム
#   RENDER_BACKEND=cuda|reference            既定 cuda
#   DRY_RUN=1                                コマンドを表示するだけで実行しない
#
# 長時間ラン時の注意:
#   * config の checkpoint_interval は必ず iterations と同じにする。既定の
#     1000 のまま 300 フレーム回すと 1 フレーム 30 個のチェックポイントが残り、
#     ディスクが途中で溢れる。baseline_7k.yaml / baseline_30k.yaml は対処済み。
#   * 評価の --rgba_background は black。neu3d は背景黒で学習するので、
#     既定の white のままだと PSNR が不当に下がる。
#   * 再開すると frames_4d.json が前半を失うため、描画前に必ず作り直す。
#
# 旧スクリプト (run_both_baselines.sh ほか) は eval/archive/ に退避してある。

set -uo pipefail

# リポジトリ直下に置いても runs/ に置いても動くようにする
# (相対パスはすべてリポジトリルート起点)。
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -d "$SCRIPT_DIR/scripts" ] && [ -d "$SCRIPT_DIR/eval" ]; then
    REPO="$SCRIPT_DIR"
else
    REPO="$(cd "$SCRIPT_DIR/.." && pwd)"
fi
cd "$REPO" || exit 1

# torch も ffmpeg も 3dgs env の中にしか無い。base の python では動かない。
source ~/miniconda3/etc/profile.d/conda.sh
conda activate 3dgs

SCENE="${1:-coffee_martini}"
END_FRAME="${2:-300}"

DATA_DIR="data/neu3d/${SCENE}/converted_4d"
CONFIG="${CONFIG:-configs/neu3d/baseline_7k.yaml}"
TAG="${TAG:-${SCENE}_$(basename "$CONFIG" .yaml)}"
SCENE_DIR="output/4DGS/neu3d/${SCENE}"
# 1 ラン 1 ディレクトリ。学習出力の隣に描画・動画・可視化・評価をぶら下げる。
RUN_DIR="${RUN_DIR:-${SCENE_DIR}/${TAG}}"
RENDER_DIR="${RUN_DIR}/renders"
VIDEO_DIR="${RUN_DIR}/videos"
VIZ_DIR="${RUN_DIR}/gaussian_viz"
# CSV もラン直下。ディレクトリがランを表すのでファイル名に TAG は入れない。
RESULTS_DIR="${RUN_DIR}/results"
STAGES="${STAGES:-train render eval video viz report}"
RENDER_BACKEND="${RENDER_BACKEND:-cuda}"
DRY_RUN="${DRY_RUN:-0}"

# 既定の可視化フレームは 1 と 50 刻み。END_FRAME を超える分は出さない。
if [ -z "${VIZ_FRAMES:-}" ]; then
    VIZ_FRAMES="1"
    for f in $(seq 50 50 "$END_FRAME"); do VIZ_FRAMES="$VIZ_FRAMES $f"; done
fi

want () { case " $STAGES " in *" $1 "*) return 0 ;; *) return 1 ;; esac; }

run_step () {
    echo "+ $*"
    [ "$DRY_RUN" = "1" ] && return 0
    "$@"
}

# 完了フレーム数 = latest.pt の個数ではなく最大フレーム番号で見る
# (欠番があると個数では再開位置を誤る)。
last_completed_frame () {
    find "$RUN_DIR" -maxdepth 3 -name latest.pt 2>/dev/null \
        | sed -E 's#.*/frame_0*([0-9]+)/checkpoints/latest\.pt#\1#' \
        | sort -n | tail -1
}

failed_stages=""
note_failure () { failed_stages="$failed_stages $1"; echo "[失敗] $1"; }

# 描画が失敗したら評価と動画は成立しないので飛ばす。
RENDER_FAILED=0

echo "=============================================================="
echo " 4DGS baseline (warm-start なし)  開始 $(date '+%Y-%m-%d %H:%M:%S')"
echo " シーン  : $SCENE  (frames 1..$END_FRAME)"
echo " data    : $DATA_DIR"
echo " config  : $CONFIG"
echo " run     : $RUN_DIR"
echo " 工程    : $STAGES"
[ "$DRY_RUN" = "1" ] && echo " DRY_RUN : コマンドを表示するだけで実行しません"
echo "=============================================================="

if [ ! -d "$DATA_DIR" ]; then
    echo "[中止] データがありません: $DATA_DIR"
    echo "       eval/convert_neu3d.py / eval/undistort_neu3d.py で変換してください。"
    exit 1
fi
if [ ! -f "$CONFIG" ]; then
    echo "[中止] 設定ファイルがありません: $CONFIG"
    exit 1
fi

# ------------------------------------------------------------------ 1. 学習
if want train; then
    # 各フレーム出力は exist_policy: error なので、未完了フレームから始める。
    START=1
    LAST="$(last_completed_frame)"
    if [ -n "$LAST" ]; then
        START=$((LAST + 1))
        echo "既存の完了フレーム ${LAST} を検出 -> frame ${START} から再開"
    fi
    echo ""
    echo "############################################################"
    echo "# 学習 frame ${START}..${END_FRAME}  $(date '+%m-%d %H:%M')"
    echo "############################################################"
    if [ "$START" -gt "$END_FRAME" ]; then
        echo "frame ${END_FRAME} まで完了済み。学習をスキップします。"
    else
        run_step python eval/warmstart_trainer.py \
            --source_path "$DATA_DIR" \
            --output_dir  "$RUN_DIR" \
            --config      "$CONFIG" \
            --no_warmstart \
            --start_frame "$START" --end_frame "$END_FRAME" \
            --image-directory images --render-backend "$RENDER_BACKEND" \
            || { note_failure "train"; echo "学習が失敗したので後段は行いません。"; exit 1; }
    fi
fi

DONE_COUNT=$(find "$RUN_DIR" -maxdepth 3 -name latest.pt 2>/dev/null | wc -l)
echo ""
echo "完了フレーム: ${DONE_COUNT}/${END_FRAME}"
if [ "$DONE_COUNT" -eq 0 ] && [ "$DRY_RUN" != "1" ]; then
    echo "[中止] 完了フレームがありません。"
    exit 1
fi

# ------------------------------------------------------------------ 2. 描画
if want render; then
    echo ""
    echo "############################################################"
    echo "# 描画  $(date '+%m-%d %H:%M')"
    echo "############################################################"
    # 再開していると frames_4d.json がその実行で回した分しか持たないので、
    # 描画前に必ず作り直す。
    run_step python eval/rebuild_manifest_4d.py \
        --run_dir "$RUN_DIR" --source_path "$DATA_DIR" \
        --frame_count "$END_FRAME" \
        || note_failure "manifest"
    run_step mkdir -p "$RENDER_DIR"
    if run_step python eval/render_4d.py \
        --run_dir "$RUN_DIR" --data_dir "$DATA_DIR" --out_dir "$RENDER_DIR" \
        --split test --render-backend "$RENDER_BACKEND" --end_frame "$END_FRAME"
    then
        [ "$DRY_RUN" != "1" ] && \
            echo "PNG: $(find "$RENDER_DIR/renders" -name '*.png' 2>/dev/null | wc -l) 枚"
    else
        note_failure "render"
        RENDER_FAILED=1
        echo "描画が失敗したので評価・動画は行いません。"
    fi
fi

# ------------------------------------------------------------------ 3. 評価
if want eval && [ "$RENDER_FAILED" = "0" ]; then
    echo ""
    echo "############################################################"
    echo "# 評価  $(date '+%m-%d %H:%M')"
    echo "############################################################"
    run_step mkdir -p "$RESULTS_DIR"
    # neu3d は背景黒で学習しているので rgba_background も black に揃える。
    run_step python eval/evaluate.py \
        --dataset neu3d \
        --render_dir "$RENDER_DIR/renders" --gt_dir "$RENDER_DIR/gt" \
        --output_csv "$RESULTS_DIR/metrics.csv" \
        --json_out "$RESULTS_DIR/summary.json" \
        --device cuda --rgba_background black \
        || note_failure "eval"
    run_step python eval/evaluate_per_frame.py \
        --render_dir "$RENDER_DIR/renders" --gt_dir "$RENDER_DIR/gt" \
        --output_csv "$RESULTS_DIR/per_frame.csv" \
        --dataset neu3d --device cuda --rgba_background black --block 10 \
        || note_failure "eval/per_frame"
    # baseline は毎フレーム独立なので数は増えないはずだが、比較用に必ず残す。
    run_step python eval/gaussian_count_trend.py \
        --run_dir "$RUN_DIR" --block 10 \
        --output_csv "$RESULTS_DIR/gaussian_counts.csv" \
        || note_failure "eval/gaussian_counts"
fi

# ------------------------------------------------------------------ 4. 動画
if want video && [ "$RENDER_FAILED" = "0" ]; then
    echo ""
    echo "############################################################"
    echo "# 動画  $(date '+%m-%d %H:%M')"
    echo "############################################################"
    # カメラごとに解像度が数 px 違うので、縦積みは make_videos_4d.py 側で
    # 幅を揃えてから積んでいる。ffmpeg は 3dgs env のものを使う。
    run_step mkdir -p "$VIDEO_DIR"
    run_step python eval/make_videos_4d.py \
        --render_root "$RENDER_DIR" --out_dir "$VIDEO_DIR" --fps 30 --crf 18 \
        || note_failure "video"
fi

# -------------------------------------------------- 5. Gaussian 可視化
if want viz; then
    echo ""
    echo "############################################################"
    echo "# Gaussian 可視化  $(date '+%m-%d %H:%M')"
    echo "############################################################"
    run_step mkdir -p "$VIZ_DIR"
    viz_done=""
    for frame in $VIZ_FRAMES; do
        printf -v padded "%04d" "$frame"
        if [ -d "$RUN_DIR/frame_${padded}/checkpoints" ] || [ "$DRY_RUN" = "1" ]; then
            if run_step python eval/visualize_gaussians.py \
                --ckpt_dir "$RUN_DIR/frame_${padded}" \
                --out_dir "$VIZ_DIR" --frame "$frame"
            then
                viz_done="$viz_done $frame"
            fi
        else
            echo "[スキップ] frame ${padded}: チェックポイント無し"
        fi
    done
    if [ -n "$viz_done" ]; then
        run_step python eval/gaussian_viz_report.py --viz_dir "$VIZ_DIR" \
            --frames $viz_done --title "Gaussian 可視化 — ${TAG}" \
            || note_failure "viz/report"
    fi
fi

# ------------------------------------------------- 6. eval.md 生成
if want report; then
    echo ""
    echo "############################################################"
    echo "# eval.md 生成  $(date '+%m-%d %H:%M')"
    echo "############################################################"
    # 人が書く節 (定性的評価 / AIによる初見) は既存 eval.md から引き継がれる。
    # 比較表に別ランを並べるときは COMPARE="<run_dir> <run_dir>" を渡す。
    report_command=(
        python eval/make_eval_report.py
        --run_dir "$RUN_DIR"
        --render_backend "$RENDER_BACKEND"
        --command "CONFIG=$CONFIG runs/4DGS_baseline.sh $SCENE $END_FRAME"
    )
    for other in ${COMPARE:-}; do
        report_command+=(--compare "$other")
    done
    run_step "${report_command[@]}" || note_failure "report"
fi

echo ""
echo "=============================================================="
echo " 完了 $(date '+%Y-%m-%d %H:%M:%S')   タグ: ${TAG}"
echo " 失敗した工程:${failed_stages:- なし}"
echo "=============================================================="
if [ "$DRY_RUN" != "1" ]; then
    du -sh "$RUN_DIR" "$RENDER_DIR" "$VIDEO_DIR" 2>/dev/null
    df -h . | tail -1
fi

[ -z "$failed_stages" ]
