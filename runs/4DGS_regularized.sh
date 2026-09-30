#!/usr/bin/env bash
# 4DGS_regularized.sh — Neu3D の動的シーンを Dynamic 3D Gaussians の正則化つきで回す。
# 前フレームの結果を次フレームの初期値にするのは 4DGS_warmstart.sh と同じで、
# それに加えて frame 2 以降で局所剛性 / 回転類似性 / 長期等長性 / 色の一貫性の
# 正則化をかける (scripts/train_4d_regularized.py,
# extensions/4dgs/dynamic_regularization.py を参照)。
#
#   runs/4DGS_regularized.sh                       # coffee_martini / 300 フレーム
#   runs/4DGS_regularized.sh cook_spinach 100
#   setsid nohup runs/4DGS_regularized.sh > logs/4dgs_regularized.log 2>&1 &
#
# tmux / screen は入っていないので、長時間ジョブは setsid nohup で detach する。
#
# 工程: 学習 -> マニフェスト再構築 -> test view 描画 -> 評価 -> 動画 ->
#       Gaussian 可視化 -> eval.md 生成。学習以外は 4DGS_warmstart.sh と同じ。
#
# frame 1 について:
#   正則化は近傍グラフでガウシアン集合を固定するので、frame 2 以降は密度制御を
#   止める必要がある。frame 1 だけは密度制御ありでしっかり学習したいので、
#   本スクリプトは frame 1 を学習しない。既存ランの frame_0001 (既定は
#   <scene>_baseline_30k) を <RUN_DIR>/frame_0001 にシンボリックリンクして
#   frame 2 からの引き継ぎ元にし、描画・評価でも 1 フレーム目として扱う。
#   frame 1 の結果は比較相手のランと同一になる。
#
# 環境変数:
#   DATA_DIR="data/dynamic/neu3d/<SCENE>/converted_4d"  変換済みデータ
#   CONFIG="configs/neu3d/dynamic_regularization.yaml"  学習設定 (frame 2 以降)
#   FRAME1_RUN="output/4DGS/neu3d/<SCENE>/<SCENE>_baseline_30k"
#                                            frame_0001/checkpoints/latest.pt を持つラン
#   TAG="<scene>_<config 名>"                出力ディレクトリの名前
#   RUN_DIR="output/4DGS/neu3d/<SCENE>/<TAG>"  ラン一式 (学習/描画/動画/可視化/評価)
#   STAGES="train render eval video viz report"  実行する工程 (既定は全部)
#   COMPARE="<run_dir> ..."                  eval.md の比較表に並べる別ラン
#   VIZ_FRAMES="1 50 100 ..."                Gaussian 可視化するフレーム
#   RENDER_BACKEND=cuda|reference            既定 cuda
#   DRY_RUN=1                                コマンドを表示するだけで実行しない
#
# 正則化固有の注意:
#   * 途中で止まったら同じコマンドを再実行すれば、完了済みの最終フレームの
#     latest.pt を --carry-over-checkpoint に渡して続きから再開する。近傍グラフと
#     速度の起点は <RUN_DIR>/dynamic_regularization/ に残っているので、中断なしと
#     同じ結果になる。このディレクトリは消さないこと。
#   * neighbor_weight_lambda (既定 2000) は論文のメートル単位のシーン向けの値。
#     COLMAP スケールでは近傍重みが 0 に潰れて正則化が効かないことがあるので、
#     本番の前に短いフレーム数で <RUN_DIR>/frame_0002/train_log.jsonl の
#     loss_reg_* が 0 に張り付いていないか確認する。
#   * eval.md の設定欄は先頭フレーム (= frame 1 の元ランの config) を読むので、
#     正則化の設定は <RUN_DIR>/config.yaml の方を見ること。
#
# 長時間ラン時の注意:
#   * 評価の --rgba-background は black。neu3d は背景黒で学習するので、
#     既定の white のままだと PSNR が不当に下がる。
#   * 再開すると frames_4d.json が前半を失うため、描画前に必ず作り直す。

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

DATA_DIR="${DATA_DIR:-data/dynamic/neu3d/${SCENE}/converted_4d}"
CONFIG="${CONFIG:-configs/neu3d/dynamic_regularization.yaml}"
TAG="${TAG:-${SCENE}_$(basename "$CONFIG" .yaml)}"
SCENE_DIR="output/4DGS/neu3d/${SCENE}"
FRAME1_RUN="${FRAME1_RUN:-${SCENE_DIR}/${SCENE}_baseline_30k}"
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

# latest.pt を持つフレーム番号を昇順に 1 行ずつ出す。frame_0001 はシンボリック
# リンクなので、リンクを辿らない find ではなく test -f で見る。
completed_frames () {
    for dir in "$RUN_DIR"/frame_*; do
        [ -f "$dir/checkpoints/latest.pt" ] || continue
        name="$(basename "$dir")"
        echo $((10#${name#frame_}))
    done | sort -n
}

# 完了フレーム数 = latest.pt の個数ではなく最大フレーム番号で見る
# (欠番があると個数では再開位置を誤る)。
last_completed_frame () { completed_frames | tail -1; }

failed_stages=""
note_failure () { failed_stages="$failed_stages $1"; echo "[失敗] $1"; }

# 描画が失敗したら評価と動画は成立しないので飛ばす。
RENDER_FAILED=0

FRAME1_DIR="${FRAME1_RUN}/frame_0001"

echo "=============================================================="
echo " 4DGS regularized  開始 $(date '+%Y-%m-%d %H:%M:%S')"
echo " シーン  : $SCENE  (frames 1..$END_FRAME, frame 1 は ${FRAME1_DIR})"
echo " data    : $DATA_DIR"
echo " config  : $CONFIG"
echo " run     : $RUN_DIR"
echo " 工程    : $STAGES"
[ "$DRY_RUN" = "1" ] && echo " DRY_RUN : コマンドを表示するだけで実行しません"
echo "=============================================================="

if [ ! -d "$DATA_DIR" ]; then
    echo "[中止] データがありません: $DATA_DIR"
    echo "       data/dynamic/neu3d/convert_neu3d.py / undistort_neu3d.py で変換してください。"
    exit 1
fi
if [ ! -f "$CONFIG" ]; then
    echo "[中止] 設定ファイルがありません: $CONFIG"
    exit 1
fi

# ------------------------------------------------------------------ 1. 学習
if want train; then
    # frame 1 は既存ランのものをリンクで借りる (上の「frame 1 について」)。
    if [ ! -e "$RUN_DIR/frame_0001" ]; then
        if [ ! -f "$FRAME1_DIR/checkpoints/latest.pt" ] && [ "$DRY_RUN" != "1" ]; then
            echo "[中止] frame 1 のチェックポイントがありません: $FRAME1_DIR/checkpoints/latest.pt"
            echo "       先に runs/4DGS_baseline.sh などで frame 1 を学習するか、FRAME1_RUN を指定してください。"
            exit 1
        fi
        run_step mkdir -p "$RUN_DIR"
        # リンク先は絶対パスにして、RUN_DIR の置き場所に依存しないようにする。
        run_step ln -s "$(cd "$(dirname "$FRAME1_DIR")" 2>/dev/null && pwd)/frame_0001" \
            "$RUN_DIR/frame_0001" \
            || { note_failure "train"; exit 1; }
    fi

    # 各フレーム出力は exist_policy: error なので、未完了フレームから始める。
    LAST="$(last_completed_frame)"
    [ -z "$LAST" ] && LAST=1   # DRY_RUN でリンクをまだ作っていないとき
    START=$((LAST + 1))
    echo "完了済みの最終フレーム ${LAST} -> frame ${START} から学習"
    echo ""
    echo "############################################################"
    echo "# 学習 frame ${START}..${END_FRAME}  $(date '+%m-%d %H:%M')"
    echo "############################################################"
    if [ "$START" -gt "$END_FRAME" ]; then
        echo "frame ${END_FRAME} まで完了済み。学習をスキップします。"
    else
        printf -v carry "%s/frame_%04d/checkpoints/latest.pt" "$RUN_DIR" "$LAST"
        if [ ! -f "$carry" ] && [ "$DRY_RUN" != "1" ]; then
            note_failure "train"
            echo "[中止] 引き継ぎ元が見つかりません: $carry"
            exit 1
        fi
        echo "引き継ぎ元: $carry"
        # 近傍グラフと速度の起点は <RUN_DIR>/dynamic_regularization/ から読むので、
        # 再開でも --regularization-state は要らない。
        run_step python scripts/train_4d_regularized.py \
            --data "$DATA_DIR" \
            --config "$CONFIG" \
            --output "$RUN_DIR" \
            --start-frame "$START" --end-frame "$END_FRAME" \
            --carry-over-checkpoint "$carry" \
            --image-directory images --render-backend "$RENDER_BACKEND" \
            --disable-training-evaluation \
            || { note_failure "train"; echo "学習が失敗したので後段は行いません。"; exit 1; }
    fi
fi

DONE_COUNT=$(completed_frames | wc -l)
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
    run_step python scripts/rebuild_manifest_4d.py \
        --run_dir "$RUN_DIR" --source_path "$DATA_DIR" \
        --frame_count "$END_FRAME" \
        || note_failure "manifest"
    run_step mkdir -p "$RENDER_DIR"
    if run_step python scripts/rendering/render_4d.py \
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
    run_step python scripts/evaluate.py \
        --dataset neu3d \
        --render-dir "$RENDER_DIR/renders" --gt-dir "$RENDER_DIR/gt" \
        --output-csv "$RESULTS_DIR/metrics.csv" \
        --json-out "$RESULTS_DIR/summary.json" \
        --per-frame-csv "$RESULTS_DIR/per_frame.csv" --block 10 \
        --device cuda --rgba-background black \
        || note_failure "eval"
    # 正則化ランは frame 2 以降でガウシアン数が変わらないはず。念のため残す。
    run_step python eval/plot/plot_gaussian_count.py \
        --run "$RUN_DIR" --block 10 \
        --output_csv "$RESULTS_DIR/gaussian_counts.csv" \
        || note_failure "eval/gaussian_counts"
fi

# ------------------------------------------------------------------ 4. 動画
if want video && [ "$RENDER_FAILED" = "0" ]; then
    echo ""
    echo "############################################################"
    echo "# 動画  $(date '+%m-%d %H:%M')"
    echo "############################################################"
    # カメラごとに 1 本 (<camera>.mp4)。ffmpeg は 3dgs env のものを使う。
    run_step mkdir -p "$VIDEO_DIR"
    run_step python scripts/rendering/make_video.py \
        --render-root "$RENDER_DIR" --out-dir "$VIDEO_DIR" --fps 30 --crf 18 \
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
            if run_step python eval/visualization/visualize_gaussians.py \
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
        run_step python eval/visualization/gaussian_viz_report.py --viz_dir "$VIZ_DIR" \
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
        python eval/report/make_eval_report.py
        --run_dir "$RUN_DIR" --data_dir "$DATA_DIR"
        --render_backend "$RENDER_BACKEND"
        --command "CONFIG=$CONFIG FRAME1_RUN=$FRAME1_RUN runs/4DGS_regularized.sh $SCENE $END_FRAME"
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
