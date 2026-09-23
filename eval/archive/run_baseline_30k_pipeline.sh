#!/usr/bin/env bash
# eval/run_baseline_30k_pipeline.sh
# baseline 30,000 iter の学習完了を待ってから、フェーズ 2-4 を続けて回す。
#
#   フェーズ 2: 全 300 フレームのレンダリング (900 枚) と PSNR/D-SSIM/LPIPS 評価
#   フェーズ 3: カメラ別 pred 動画 / GT 比較動画 / 全カメラ縦積み動画
#   フェーズ 4: 要所フレームの Gaussian 可視化と比較 HTML
#
# 学習プロセス (eval/warmstart_trainer.py) とは別に detach して走らせる想定:
#   setsid nohup eval/run_baseline_30k_pipeline.sh > logs/pipeline_30k.log 2>&1 &
#
# 学習が落ちて途中で止まっていた場合は、そこまでのフレームで処理を進める。
# frames_4d.json は再開時に前半が消えることがあるので、必ず作り直してから使う
# (eval/rebuild_manifest_4d.py の docstring を参照)。

set -uo pipefail

REPO=/home/chihiro-tanaka/Git/3DGaussianSplatting
cd "$REPO"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate 3dgs

RUN_DIR=output/4DGS/neu3d/baseline_30k
DATA_DIR=data/neu3d/coffee_martini/converted_4d
RENDER_DIR=output/4DGS/neu3d/renders/baseline_30k
VIDEO_DIR=output/4DGS/neu3d/videos/baseline_30k
VIZ_DIR=eval/gaussian_viz/baseline_30k
RESULT_CSV=eval/results/baseline_30k.csv
TOTAL_FRAMES=300
VIZ_FRAMES="1 50 100 150 200 250 300"

echo "=============================================================="
echo " baseline 30k パイプライン開始: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=============================================================="

# ---------------------------------------------------------------- 学習待ち
# 学習ドライバが生きている間は待つ。pgrep は「このリポジトリの
# warmstart_trainer.py で output/4DGS/neu3d/baseline_30k に書いているもの」だけを見る。
# 先頭を [w] と括るのは自分自身にマッチさせないため: このスクリプトを
# 引数付きで起動したシェルのコマンド行にもパターン文字列が載るので、
# 素の "warmstart_trainer" だと永久に「学習中」と誤判定して待ち続ける。
while pgrep -f "[w]armstart_trainer\.py.*baseline_30k" > /dev/null; do
    done_count=$(find "$RUN_DIR" -maxdepth 3 -name latest.pt 2>/dev/null | wc -l)
    echo "[$(date '+%m-%d %H:%M')] 学習中... 完了フレーム ${done_count}/${TOTAL_FRAMES}"
    sleep 600
done

DONE=$(find "$RUN_DIR" -maxdepth 3 -name latest.pt 2>/dev/null | wc -l)
echo ""
echo "学習プロセスは終了しました。完了フレーム: ${DONE}/${TOTAL_FRAMES}"
if [ "$DONE" -eq 0 ]; then
    echo "[中止] 完了フレームが 1 つもありません。logs/baseline_30k.log を確認してください。"
    exit 1
fi
if [ "$DONE" -lt "$TOTAL_FRAMES" ]; then
    echo "[警告] 300 フレームに届いていません。ある分だけで後続を進めます。"
fi

# ------------------------------------------------- マニフェストを作り直す
echo ""
echo "--- マニフェスト再構築 ---"
python eval/rebuild_manifest_4d.py \
    --run_dir "$RUN_DIR" \
    --source_path "$DATA_DIR" \
    --frame_count "$TOTAL_FRAMES" || exit 1

# ------------------------------------------------------ フェーズ 2: 描画
echo ""
echo "--- フェーズ 2-1: レンダリング ---"
mkdir -p "$RENDER_DIR"
python eval/render_4d.py \
    --run_dir  "$RUN_DIR" \
    --data_dir "$DATA_DIR" \
    --out_dir  "$RENDER_DIR" \
    --split test \
    --render-backend cuda || exit 1

RENDERED=$(find "$RENDER_DIR/renders" -name "*.png" | wc -l)
echo "レンダリング済み PNG: ${RENDERED} 枚 (期待値 $((TOTAL_FRAMES * 3)))"

# ------------------------------------------------------ フェーズ 2-2: 評価
echo ""
echo "--- フェーズ 2-2: 画質評価 ---"
mkdir -p eval/results
# rgba_background は学習 config の data.rgba_background / rendering.background と
# 揃える必要がある (neu3d/base.yaml はどちらも黒)。既定の white のままだと
# 背景が食い違って PSNR が不当に下がる。
python eval/evaluate.py \
    --dataset neu3d \
    --render_dir "$RENDER_DIR/renders" \
    --gt_dir     "$RENDER_DIR/gt" \
    --output_csv "$RESULT_CSV" \
    --device cuda \
    --rgba_background black || exit 1

python eval/summarize_camera_metrics.py \
    --csv "$RESULT_CSV" \
    --json_out eval/results/baseline_30k_summary.json

# evaluate.py の CSV はファイル名 (cam00.png) しか持たず、300 フレームすべてが
# 同じ名前なのでフレーム別の推移が出せない。フレーム番号つきで取り直す。
echo ""
echo "--- フェーズ 2-3: フレーム別指標と 10 フレームブロック推移 ---"
python eval/evaluate_per_frame.py \
    --render_dir "$RENDER_DIR/renders" \
    --gt_dir     "$RENDER_DIR/gt" \
    --output_csv eval/results/baseline_30k_per_frame.csv \
    --dataset neu3d --device cuda --rgba_background black --block 10

echo ""
echo "--- フェーズ 2-4: Gaussian 数の推移 ---"
python eval/gaussian_count_trend.py \
    --run_dir "$RUN_DIR" --block 10 \
    --output_csv eval/results/baseline_30k_gaussian_counts.csv

# ------------------------------------------------------ フェーズ 3: 動画
echo ""
echo "--- フェーズ 3: 動画生成 ---"
mkdir -p "$VIDEO_DIR"
python eval/make_videos_4d.py \
    --render_root "$RENDER_DIR" \
    --out_dir     "$VIDEO_DIR" \
    --fps 30 --crf 18

# ------------------------------------------- フェーズ 4: Gaussian 可視化
echo ""
echo "--- フェーズ 4: Gaussian 可視化 ---"
mkdir -p "$VIZ_DIR"
VIZ_DONE=""
for frame in $VIZ_FRAMES; do
    printf -v padded "%04d" "$frame"
    ckpt_dir="$RUN_DIR/frame_${padded}"
    if [ -d "$ckpt_dir/checkpoints" ]; then
        python eval/visualize_gaussians.py \
            --ckpt_dir "$ckpt_dir" \
            --out_dir  "$VIZ_DIR" \
            --frame    "$frame" && VIZ_DONE="$VIZ_DONE $frame"
    else
        echo "[スキップ] $ckpt_dir にチェックポイントがありません"
    fi
done

if [ -n "$VIZ_DONE" ]; then
    python eval/gaussian_viz_report.py \
        --viz_dir "$VIZ_DIR" \
        --frames $VIZ_DONE
fi

echo ""
echo "=============================================================="
echo " 完了: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=============================================================="
echo "  学習       : $RUN_DIR (${DONE} フレーム)"
echo "  レンダリング: $RENDER_DIR (${RENDERED} 枚)"
echo "  評価       : $RESULT_CSV"
echo "  フレーム別 : eval/results/baseline_30k_per_frame.csv"
echo "  Gaussian数 : eval/results/baseline_30k_gaussian_counts.csv"
echo "  動画       : $VIDEO_DIR"
echo "  可視化     : $VIZ_DIR/comparison.html"
du -sh "$RUN_DIR" "$RENDER_DIR" "$VIDEO_DIR" 2>/dev/null
