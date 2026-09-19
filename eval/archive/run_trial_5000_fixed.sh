#!/usr/bin/env bash
# eval/run_trial_5000_fixed.sh
# configs/neu3d/trial_5000_fixed.yaml (densify_until 15000 -> 4500) で 5000 iter /
# 10 frames の baseline を学習し直し、描画・評価まで通す。
#
#   cd /home/chihiro-tanaka/Git/3DGaussianSplatting
#   setsid nohup bash eval/run_trial_5000_fixed.sh > logs/trial_5000_fixed.log 2>&1 &
#
# なぜ作り直すか:
#   configs/neu3d/trial_5000.yaml は iterations=5000 なのに densify_until=15000 で、
#   末尾の収束期間が無いまま densification が最終 iter まで走る。4000/7000 は
#   いずれも N-500 なので、5000 だけが別条件だった (4000:24.50 > 7000:23.71 >
#   5000:22.99 dB)。densify_until だけを 4500 に直して条件を揃える。
#
# 30k 学習との並行について:
#   学習結果は wall-clock ではなく iteration で決まるので、GPU 競合があっても
#   PSNR / Gaussian 数は汚染されない。汚れるのは速度計測だけ。よって並行で
#   流し、30k 側の減速幅だけを実測して記録する。
#   supervisor (run_both_baselines.sh) は
#   pgrep -f "[w]armstart_trainer\.py.*baseline_30k" で 30k だけを待つので、
#   output_dir に baseline_30k を含まない本ジョブはキューに影響しない。

set -uo pipefail

REPO=/home/chihiro-tanaka/Git/3DGaussianSplatting
cd "$REPO" || exit 1

source ~/miniconda3/etc/profile.d/conda.sh
conda activate 3dgs

DATA_DIR=data/neu3d/coffee_martini/converted_4d
CONFIG=configs/neu3d/trial_5000_fixed.yaml
RUN_DIR=output/4DGS/neu3d/baseline_neu3d_5000_fixed_trial
RUN_TAG=trial_baseline_5000_fixed
OUT_DIR=output/4DGS/neu3d/renders/${RUN_TAG}
LOG_DIR=eval/logs
TRAIN_ROOT=output/4DGS/neu3d/baseline_30k
FLAT_ITER=16000          # 30k 側 densify 終了後の平坦域だけで比較する

mkdir -p "$LOG_DIR" eval/results "$OUT_DIR"

echo "========================================="
echo " 5000_fixed 再学習  $(date '+%Y-%m-%d %H:%M:%S')"
echo " config : ${CONFIG}"
echo " out    : ${RUN_DIR}"
echo "========================================="
echo ""
echo "--- neu3d/trial_5000.yaml との差分 ---"
diff configs/neu3d/trial_5000.yaml "$CONFIG"
echo ""

nvidia-smi --query-gpu=timestamp,memory.used,memory.free,utilization.gpu \
    --format=csv -l 10 > "${LOG_DIR}/${RUN_TAG}_vram.log" 2>&1 &
VRAM_PID=$!
trap 'kill ${VRAM_PID} 2>/dev/null || true' EXIT

# --- 学習 ------------------------------------------------------------------
TRAIN_START=$(date +%s.%N)
echo "--- 学習 開始 $(date '+%H:%M:%S') ---"
python eval/warmstart_trainer.py \
    --source_path "$DATA_DIR" \
    --output_dir  "$RUN_DIR" \
    --config      "$CONFIG" \
    --no_warmstart \
    --start_frame 1 --end_frame 10 \
    --image-directory images --render-backend cuda \
    2>&1 | tee "${LOG_DIR}/${RUN_TAG}_train.log"
TRAIN_RC=${PIPESTATUS[0]}
TRAIN_END=$(date +%s.%N)
echo "--- 学習 終了 $(date '+%H:%M:%S')  (rc=${TRAIN_RC}) ---"
kill ${VRAM_PID} 2>/dev/null || true

if [ "$TRAIN_RC" -ne 0 ]; then
    echo "[中止] 学習が失敗しました。描画・評価はスキップします。"
    exit 1
fi
echo "完了フレーム: $(find "$RUN_DIR" -maxdepth 3 -name latest.pt | wc -l)/10"

# --- 30k 側への影響を実測 --------------------------------------------------
# 本ジョブは数十分走り 30k の複数フレームをまたぐので、既存スクリプトの
# 単一フレーム計測ではなく、全フレームの平坦域サンプルを絶対時刻に直して
# 学習ウィンドウの前/中/後に振り分ける。
echo ""
echo "--- 30k 学習への影響 ---"
TRAIN_ROOT="$TRAIN_ROOT" WIN_START="$TRAIN_START" WIN_END="$TRAIN_END" \
FLAT_ITER="$FLAT_ITER" python3 <<'PYEOF'
import json, os, glob, statistics, datetime

root = os.environ["TRAIN_ROOT"]
ws, we = float(os.environ["WIN_START"]), float(os.environ["WIN_END"])
flat = int(os.environ["FLAT_ITER"])

pre, during, post = [], [], []
for cfg in sorted(glob.glob(os.path.join(root, "frame_*", "config.yaml"))):
    d = os.path.dirname(cfg)
    log = os.path.join(d, "train_log.jsonl")
    if not os.path.isfile(log):
        continue
    start = os.stat(cfg).st_mtime          # 学習開始の絶対時刻
    rows = []
    with open(log) as f:
        for line in f:
            try:
                r = json.loads(line)
            except Exception:
                continue                    # 追記中の行末が欠けることがある
            if r.get("iteration", 0) >= flat:
                rows.append(r)
    for a, b in zip(rows, rows[1:]):
        ta, tb = start + a["elapsed_seconds"], start + b["elapsed_seconds"]
        delta = b["elapsed_seconds"] - a["elapsed_seconds"]
        if tb <= ws:   pre.append(delta)
        elif ta >= we: post.append(delta)
        else:          during.append(delta)

fmt = lambda t: datetime.datetime.fromtimestamp(t).strftime("%m-%d %H:%M:%S")
print(f"  学習ウィンドウ: {fmt(ws)} 〜 {fmt(we)}  ({(we-ws)/60:.1f} 分)")

def show(label, xs):
    if not xs:
        print(f"  {label}: 記録なし"); return None
    m = statistics.mean(xs)
    s = statistics.pstdev(xs) if len(xs) > 1 else 0.0
    print(f"  {label}: mean={m:.3f}s/100it  sd={s:.3f}  n={len(xs)}")
    return m

m_pre = show("並行前", pre)
m_dur = show("並行中", during)
m_post = show("並行後", post)

print()
if m_dur is None:
    print("  判定: 測定不能 — 30k 側の平坦域サンプルが window に重なりません")
else:
    base = [m for m in (m_pre, m_post) if m]
    if not base:
        print("  判定: 比較対象 (並行前/後) が無いため相対比を出せません")
    else:
        ratio = m_dur / statistics.mean(base)
        lost = (m_dur - statistics.mean(base)) * len(during) / 60.0
        print(f"  並行前後比: {ratio:.2f}x   30k の遅延 約 {lost:.1f} 分")
        if ratio < 1.15:   print("  判定: 影響なし")
        elif ratio < 2.0:  print("  判定: 軽度競合 (許容範囲)")
        else:              print("  判定: 重度競合")
PYEOF

# --- 描画 ------------------------------------------------------------------
echo ""
echo "--- 描画 開始 $(date '+%H:%M:%S') ---"
python eval/render_4d.py \
    --run_dir "$RUN_DIR" --data_dir "$DATA_DIR" --out_dir "$OUT_DIR" \
    --split test --render-backend cuda \
    2>&1 | tee "${LOG_DIR}/${RUN_TAG}_render.log"
RENDER_RC=${PIPESTATUS[0]}
if [ "$RENDER_RC" -ne 0 ]; then
    echo "[中止] 描画が失敗しました。評価はスキップします。"
    exit 1
fi
echo "PNG: $(find "$OUT_DIR/renders" -name '*.png' | wc -l) 枚"

# --- 評価 ------------------------------------------------------------------
# rgba_background は学習設定 (neu3d は黒) と揃える。既定の white だと PSNR が
# 不当に下がる。
echo ""
echo "--- 全体評価 ---"
python eval/evaluate.py \
    --dataset neu3d \
    --render_dir "$OUT_DIR/renders" --gt_dir "$OUT_DIR/gt" \
    --output_csv "eval/results/${RUN_TAG}.csv" \
    --device cuda --rgba_background black \
    2>&1 | tee "${LOG_DIR}/${RUN_TAG}_eval.log"

echo ""
echo "--- フレーム別評価 ---"
python eval/evaluate_per_frame.py \
    --render_dir "$OUT_DIR/renders" --gt_dir "$OUT_DIR/gt" \
    --output_csv "eval/results/${RUN_TAG}_per_frame.csv" \
    --dataset neu3d --device cuda --rgba_background black --block 2 \
    2>&1 | tee "${LOG_DIR}/${RUN_TAG}_eval_per_frame.log"

# --- Gaussian 数 -----------------------------------------------------------
echo ""
echo "--- Gaussian 数の推移 ---"
python eval/gaussian_count_trend.py \
    --run_dir "$RUN_DIR" --block 2 \
    --output_csv "eval/results/${RUN_TAG}_gaussian_counts.csv"

# --- trial series まとめ ---------------------------------------------------
echo ""
echo "========================================="
echo " trial baseline series"
echo "========================================="
python3 <<'PYEOF'
import csv, os, statistics
tags = [("trial_baseline_4000",       "4000      (densify_until 3500)"),
        ("trial_baseline_5000",       "5000 旧   (densify_until 15000)"),
        ("trial_baseline_5000_fixed", "5000 修正 (densify_until 4500)"),
        ("trial_baseline_7000",       "7000      (densify_until 6500)")]
print(f"  {'run':34s} {'n':>3s} {'PSNR/':>9s} {'D-SSIM/':>9s} {'LPIPS/':>9s}")
print("  " + "-" * 68)
for t, label in tags:
    p = f"eval/results/{t}.csv"
    if not os.path.isfile(p):
        print(f"  {label:34s}  (未実行)"); continue
    rows = [r for r in csv.DictReader(open(p))
            if r["filename"].lower().endswith(".png")]   # *** AVERAGE *** 行を除く
    if not rows:
        print(f"  {label:34s}  (空)"); continue
    f = lambda k: statistics.mean(float(r[k]) for r in rows)
    print(f"  {label:34s} {len(rows):3d} {f('psnr'):9.4f} {f('d_ssim'):9.4f} {f('lpips'):9.4f}")
PYEOF

echo ""
echo "========================================="
echo " 完了 $(date '+%Y-%m-%d %H:%M:%S')"
echo " 結果 : eval/results/${RUN_TAG}.csv"
echo "        eval/results/${RUN_TAG}_per_frame.csv"
echo "        eval/results/${RUN_TAG}_gaussian_counts.csv"
echo "========================================="
