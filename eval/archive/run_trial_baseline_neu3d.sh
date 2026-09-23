#!/usr/bin/env bash
# eval/run_trial_baseline_neu3d.sh
# baseline_neu3d_trial (5000 iter / 10 frames) の描画・評価を、baseline_30k の
# 学習と並行して流し、学習側への影響を実測する。
#
#   cd /home/chihiro-tanaka/Git/3DGaussianSplatting
#   source ~/miniconda3/etc/profile.d/conda.sh && conda activate 3dgs
#   bash eval/run_trial_baseline_neu3d.sh
#
# 計測の考え方:
#   train_log.jsonl は 100 iter ごとに elapsed_seconds を記録するが絶対時刻を
#   持たない。frame_NNNN/config.yaml の mtime が学習開始時刻と厳密に一致する
#   ので (実測誤差 1ms 未満)、絶対時刻 = mtime(config.yaml) + elapsed_seconds
#   で復元して描画ウィンドウと突き合わせる。
#   densify_until_iteration=15000 までは Gaussian 数が増え続け速度が 2.3→2.7s
#   と動くので、比較は iter>=16000 の平坦域だけで行う (mean 2.769 / sd 0.084)。

set -uo pipefail

REPO=/home/chihiro-tanaka/Git/3DGaussianSplatting
cd "$REPO" || exit 1

RUN_DIR=output/4DGS/neu3d/baseline_neu3d_trial
DATA_DIR=data/neu3d/coffee_martini/converted_4d
RUN_TAG=trial_baseline_5000
OUT_DIR=output/4DGS/neu3d/renders/${RUN_TAG}
LOG_DIR=eval/logs
TRAIN_ROOT=output/4DGS/neu3d/baseline_30k
FLAT_ITER=16000          # densify 終了後の平坦域
THR_OK=3.0               # mean+3sd = 3.02
THR_NG=5.5

mkdir -p "$LOG_DIR" eval/results "$OUT_DIR"

echo "========================================="
echo " trial 描画 + 競合実測   $(date '+%Y-%m-%d %H:%M:%S')"
echo " RUN_TAG: ${RUN_TAG}"
echo "========================================="

# --- 実行中フレームを動的に特定する ---------------------------------------
# checkpoints/ は完了時にしか作られないので、「config.yaml はあるが
# checkpoints/latest.pt が無い」フレームが現在学習中のもの。
current_frame () {
    local d
    for d in $(ls -d ${TRAIN_ROOT}/frame_* 2>/dev/null | sort); do
        if [ -f "$d/config.yaml" ] && [ ! -f "$d/checkpoints/latest.pt" ]; then
            echo "$d"; return 0
        fi
    done
    return 1
}

iter_of () {  # $1=frame_dir  -> 最新 iteration (読めなければ -1)
    tail -1 "$1/train_log.jsonl" 2>/dev/null \
        | python3 -c 'import json,sys
try: print(json.load(sys.stdin)["iteration"])
except Exception: print(-1)' 2>/dev/null || echo -1
}

echo ""
echo "--- 学習中フレームが iter ${FLAT_ITER} を超えるまで待機 ---"
FRAME=""
while :; do
    FRAME=$(current_frame) || { echo "  [中止] 学習中のフレームがありません (学習が終了済み?)"; exit 1; }
    IT=$(iter_of "$FRAME")
    if [ "$IT" -ge "$FLAT_ITER" ] 2>/dev/null; then
        # 描画 (約100s) が終わる前にフレームが完了すると測定できない。
        # 30000 まで残り 1500 iter (約42s) を切っていたら次フレームを待つ。
        if [ "$IT" -ge 28500 ]; then
            echo "  $(date '+%H:%M:%S')  $(basename "$FRAME") iter ${IT} — 完了間近なので次フレームを待ちます"
        else
            echo "  $(date '+%H:%M:%S')  $(basename "$FRAME") iter ${IT} → 描画開始"
            break
        fi
    else
        echo "  $(date '+%H:%M:%S')  $(basename "$FRAME") iter ${IT} / ${FLAT_ITER} ..."
    fi
    sleep 20
done
MEASURE_FRAME="$FRAME"
echo "  計測対象: ${MEASURE_FRAME}"

# --- VRAM モニタ (trap で必ず落とす) --------------------------------------
nvidia-smi --query-gpu=timestamp,memory.used,memory.free,utilization.gpu \
    --format=csv -l 5 > "${LOG_DIR}/${RUN_TAG}_vram.log" 2>&1 &
VRAM_PID=$!
trap 'kill ${VRAM_PID} 2>/dev/null || true' EXIT
echo "  VRAM monitor PID: ${VRAM_PID}"

# --- 描画 ------------------------------------------------------------------
RENDER_START_EPOCH=$(date +%s.%N)
echo ""
echo "--- 描画  開始 $(date '+%H:%M:%S') ---"
python eval/render_4d.py \
    --run_dir  "$RUN_DIR" \
    --data_dir "$DATA_DIR" \
    --out_dir  "$OUT_DIR" \
    --split test --render-backend cuda \
    2>&1 | tee "${LOG_DIR}/${RUN_TAG}_render.log"
RENDER_RC=${PIPESTATUS[0]}
RENDER_END_EPOCH=$(date +%s.%N)
echo "--- 描画  終了 $(date '+%H:%M:%S')  (rc=${RENDER_RC}) ---"
kill ${VRAM_PID} 2>/dev/null || true

if [ "$RENDER_RC" -ne 0 ]; then
    echo "[中止] 描画が失敗しました。評価はスキップします。"
    exit 1
fi
echo "PNG: $(find "$OUT_DIR/renders" -name '*.png' | wc -l) 枚"

# --- 競合判定 --------------------------------------------------------------
echo ""
echo "--- 競合判定 ---"
MEASURE_FRAME="$MEASURE_FRAME" \
RENDER_START_EPOCH="$RENDER_START_EPOCH" RENDER_END_EPOCH="$RENDER_END_EPOCH" \
FLAT_ITER="$FLAT_ITER" THR_OK="$THR_OK" THR_NG="$THR_NG" \
python3 <<'PYEOF'
import json, os, statistics, datetime

frame = os.environ["MEASURE_FRAME"]
t0_r  = float(os.environ["RENDER_START_EPOCH"])
t1_r  = float(os.environ["RENDER_END_EPOCH"])
flat  = int(os.environ["FLAT_ITER"])
ok_thr, ng_thr = float(os.environ["THR_OK"]), float(os.environ["THR_NG"])

# 学習開始の絶対時刻。config.yaml は学習開始時に一度だけ書かれる。
start = os.stat(os.path.join(frame, "config.yaml")).st_mtime
rows = []
with open(os.path.join(frame, "train_log.jsonl")) as f:
    for line in f:
        try:
            d = json.loads(line)
        except Exception:
            continue          # 追記中の行末が欠けることがある
        if d.get("iteration", 0) >= flat:
            rows.append(d)

fmt = lambda t: datetime.datetime.fromtimestamp(t).strftime("%H:%M:%S")
print(f"  計測フレーム : {os.path.basename(frame)} (開始 {fmt(start)})")
print(f"  描画ウィンドウ: {fmt(t0_r)} 〜 {fmt(t1_r)}  ({t1_r-t0_r:.0f}s)")

if len(rows) < 2:
    print("  [判定不能] 平坦域の記録が足りません。")
    raise SystemExit(0)

pre, during, post = [], [], []
for a, b in zip(rows, rows[1:]):
    ta, tb = start + a["elapsed_seconds"], start + b["elapsed_seconds"]
    delta  = b["elapsed_seconds"] - a["elapsed_seconds"]
    if tb <= t0_r:   pre.append(delta)
    elif ta >= t1_r: post.append(delta)
    else:            during.append((b["iteration"], delta))

def show(label, xs):
    if not xs:
        print(f"  {label}: 記録なし"); return None
    m = statistics.mean(xs)
    s = statistics.pstdev(xs) if len(xs) > 1 else 0.0
    print(f"  {label}: mean={m:.3f}s/100it  sd={s:.3f}  n={len(xs)}")
    return m

m_pre  = show("描画前", pre)
if during:
    print("  描画中:")
    for it, d in during:
        tag = "影響なし" if d < ok_thr else ("軽度競合" if d < ng_thr else "重度競合")
        print(f"    iter {it:6d}: {d:6.3f}s/100it   <- {tag}")
else:
    print("  描画中: 記録なし")
m_post = show("描画後", post)

print()
if not during:
    print("  判定: 測定失敗 — 描画ウィンドウが学習フレームと重なりませんでした。")
    print("        (フレーム境界をまたいだ可能性。再実行してください)")
else:
    worst = max(d for _, d in during)
    mean_d = statistics.mean([d for _, d in during])
    print(f"  描画中 mean={mean_d:.3f}  worst={worst:.3f} s/100it")
    if worst < ok_thr:
        print("  判定: 影響なし  -> 残り4本も並行で流せます")
    elif worst < ng_thr:
        print("  判定: 軽度競合  -> 許容範囲。残り4本も並行可")
    else:
        print("  判定: 重度競合  -> 並行は見送り、学習完了後に回してください")
    if m_pre:
        print(f"  (参考) 描画前比 {mean_d/m_pre:.2f}x")
PYEOF

# --- 評価 ------------------------------------------------------------------
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

echo ""
echo "========================================="
echo " 完了 $(date '+%Y-%m-%d %H:%M:%S')"
echo " 結果 : eval/results/${RUN_TAG}.csv"
echo "        eval/results/${RUN_TAG}_per_frame.csv"
echo " VRAM : ${LOG_DIR}/${RUN_TAG}_vram.log"
echo "========================================="
