#!/usr/bin/env bash
# eval/run_both_baselines.sh
# baseline 30,000 iter と baseline 7,000 iter を順に 300 フレーム完走させ、
# それぞれについて描画 → 評価 → 動画 → Gaussian 可視化まで通し、最後に
# 2 本を突き合わせる。
#
#   setsid nohup eval/run_both_baselines.sh > logs/both_baselines.log 2>&1 &
#
# 30k の学習は既に別プロセスで走っている前提。ここでは
#   1. 30k の完走を待つ      (既存プロセスに任せる。二重起動しない)
#   2. 30k の後段を回す
#   3. 7k の学習を 300 フレーム回す
#   4. 7k の後段を回す
#   5. 比較表を出す
# の順で進める。GPU は 1 枚しか無いので学習と描画は必ず直列にする。

set -uo pipefail

REPO=/home/chihiro-tanaka/Git/3DGaussianSplatting
cd "$REPO" || exit 1

source ~/miniconda3/etc/profile.d/conda.sh
conda activate 3dgs

DATA_DIR=data/neu3d/coffee_martini/converted_4d
TOTAL_FRAMES=300
VIZ_FRAMES="1 50 100 150 200 250 300"

# ---------------------------------------------------------------------------
# 1 ラン分の後段処理。引数: <タグ> <run_dir>
# ---------------------------------------------------------------------------
postprocess () {
    local tag="$1" run_dir="$2"
    local render_dir="output/4DGS/neu3d/renders/${tag}"
    local video_dir="output/4DGS/neu3d/videos/${tag}"
    local viz_dir="eval/gaussian_viz/${tag}"

    local done_count
    done_count=$(find "$run_dir" -maxdepth 3 -name latest.pt 2>/dev/null | wc -l)
    echo ""
    echo "############################################################"
    echo "# 後段処理: ${tag}  (完了フレーム ${done_count}/${TOTAL_FRAMES})"
    echo "############################################################"
    if [ "$done_count" -eq 0 ]; then
        echo "[中止] ${tag}: 完了フレームがありません"
        return 1
    fi

    # 再開していると frames_4d.json が前半を失っているので必ず作り直す。
    python eval/rebuild_manifest_4d.py \
        --run_dir "$run_dir" --source_path "$DATA_DIR" \
        --frame_count "$TOTAL_FRAMES" || return 1

    echo ""
    echo "--- ${tag}: レンダリング ---"
    mkdir -p "$render_dir"
    python eval/render_4d.py \
        --run_dir "$run_dir" --data_dir "$DATA_DIR" --out_dir "$render_dir" \
        --split test --render-backend cuda || return 1
    echo "PNG: $(find "$render_dir/renders" -name '*.png' | wc -l) 枚"

    echo ""
    echo "--- ${tag}: 評価 (カメラ別) ---"
    mkdir -p eval/results
    # rgba_background は学習設定 (neu3d は黒) と揃える。既定の white だと
    # 背景が食い違って PSNR が不当に下がる。
    python eval/evaluate.py \
        --dataset neu3d \
        --render_dir "$render_dir/renders" --gt_dir "$render_dir/gt" \
        --output_csv "eval/results/${tag}.csv" \
        --device cuda --rgba_background black || return 1
    python eval/summarize_camera_metrics.py \
        --csv "eval/results/${tag}.csv" \
        --json_out "eval/results/${tag}_summary.json"

    echo ""
    echo "--- ${tag}: フレーム別指標と 10 フレームブロック推移 ---"
    python eval/evaluate_per_frame.py \
        --render_dir "$render_dir/renders" --gt_dir "$render_dir/gt" \
        --output_csv "eval/results/${tag}_per_frame.csv" \
        --dataset neu3d --device cuda --rgba_background black --block 10

    echo ""
    echo "--- ${tag}: Gaussian 数の推移 ---"
    python eval/gaussian_count_trend.py \
        --run_dir "$run_dir" --block 10 \
        --output_csv "eval/results/${tag}_gaussian_counts.csv"

    echo ""
    echo "--- ${tag}: 動画 ---"
    mkdir -p "$video_dir"
    python eval/make_videos_4d.py \
        --render_root "$render_dir" --out_dir "$video_dir" --fps 30 --crf 18

    echo ""
    echo "--- ${tag}: Gaussian 可視化 ---"
    mkdir -p "$viz_dir"
    local viz_done=""
    for frame in $VIZ_FRAMES; do
        local padded
        printf -v padded "%04d" "$frame"
        if [ -d "$run_dir/frame_${padded}/checkpoints" ]; then
            python eval/visualize_gaussians.py \
                --ckpt_dir "$run_dir/frame_${padded}" \
                --out_dir "$viz_dir" --frame "$frame" \
                && viz_done="$viz_done $frame"
        else
            echo "[スキップ] frame ${padded}: チェックポイント無し"
        fi
    done
    if [ -n "$viz_done" ]; then
        python eval/gaussian_viz_report.py --viz_dir "$viz_dir" \
            --frames $viz_done --title "Gaussian 可視化 — ${tag}"
    fi
    echo ""
    echo "### ${tag} 完了 $(date '+%m-%d %H:%M') ###"
}

echo "=============================================================="
echo " 両 baseline 実行  開始 $(date '+%Y-%m-%d %H:%M:%S')"
echo "=============================================================="

# --------------------------------------------------- 1. 30k の完走を待つ
# 既に走っているプロセスに任せる。[w] と括るのは自分自身にマッチさせないため。
while pgrep -f "[w]armstart_trainer\.py.*baseline_30k" > /dev/null; do
    echo "[$(date '+%m-%d %H:%M')] 30k 学習中... $(find output/4DGS/neu3d/baseline_30k -maxdepth 3 -name latest.pt 2>/dev/null | wc -l)/${TOTAL_FRAMES}"
    sleep 900
done
echo "30k 学習プロセスは終了しました。"

# --------------------------------------------------- 2. 30k の後段
postprocess "baseline_30k" "output/4DGS/neu3d/baseline_30k"

# --------------------------------------------------- 3. 7k の学習
echo ""
echo "############################################################"
echo "# 7,000 iter 学習開始 $(date '+%m-%d %H:%M')"
echo "############################################################"
# 既に一部フレームがあると exist_policy=error で落ちるので、未完了の
# 続きから始める。完了済みの最大フレーム + 1 を開始点にする。
START=1
if [ -d output/4DGS/neu3d/baseline_7k ]; then
    LAST=$(find output/4DGS/neu3d/baseline_7k -maxdepth 3 -name latest.pt 2>/dev/null \
        | sed -E 's#.*/frame_0*([0-9]+)/checkpoints/latest\.pt#\1#' \
        | sort -n | tail -1)
    if [ -n "$LAST" ]; then
        START=$((LAST + 1))
        echo "既存の完了フレーム ${LAST} を検出 -> frame ${START} から再開"
    fi
fi
if [ "$START" -le "$TOTAL_FRAMES" ]; then
    python eval/warmstart_trainer.py \
        --source_path "$DATA_DIR" \
        --output_dir  output/4DGS/neu3d/baseline_7k \
        --config      configs/neu3d/baseline_7k.yaml \
        --no_warmstart \
        --start_frame "$START" --end_frame "$TOTAL_FRAMES" \
        --image-directory images --render-backend cuda
else
    echo "7k は既に 300 フレーム完了済み"
fi

# --------------------------------------------------- 4. 7k の後段
postprocess "baseline_7k" "output/4DGS/neu3d/baseline_7k"

# --------------------------------------------------- 5. 比較
echo ""
echo "############################################################"
echo "# 30,000 iter vs 7,000 iter"
echo "############################################################"
python eval/compare_runs.py \
    --a eval/results/baseline_30k_per_frame.csv --a_label "30,000 iter" \
    --b eval/results/baseline_7k_per_frame.csv  --b_label "7,000 iter" \
    --block 10

echo ""
echo "=============================================================="
echo " 全処理完了 $(date '+%Y-%m-%d %H:%M:%S')"
echo "=============================================================="
du -sh output/4DGS/neu3d/baseline_30k output/4DGS/neu3d/baseline_7k output/4DGS/neu3d/renders output/4DGS/neu3d/videos 2>/dev/null
df -h . | tail -1
