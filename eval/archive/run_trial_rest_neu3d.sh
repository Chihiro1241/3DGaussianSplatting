#!/usr/bin/env bash
# eval/run_trial_rest_neu3d.sh
# trial 4 本 (baseline 4000/7000, warmstart 4000/7000) の描画・評価を
# baseline_30k の学習と並行して流し、学習側への影響をまとめて実測する。
#
#   cd /home/chihiro-tanaka/Git/3DGaussianSplatting
#   source ~/miniconda3/etc/profile.d/conda.sh && conda activate 3dgs
#   setsid nohup bash eval/run_trial_rest_neu3d.sh > logs/trial_rest.log 2>&1 &
#
# 5000 iter 1 本での実測は 描画前 2.649 -> 描画中 2.732 s/100it (1.03x) で
# 影響なし。4 本まとめても描画は 2-3 分の見込みなので、平坦域
# (iter>=16000, densify 終了後) 1 フレームの中に収める。

set -uo pipefail

REPO=/home/chihiro-tanaka/Git/3DGaussianSplatting
cd "$REPO" || exit 1

DATA_DIR=data/neu3d/coffee_martini/converted_4d
LOG_DIR=eval/logs
TRAIN_ROOT=output/4DGS/neu3d/baseline_30k
FLAT_ITER=16000
# 描画バッチ (2-3 分) が平坦域に収まるよう、入口を絞る。
# iter 22000 なら 30000 まで 8000 iter = 約 220 秒残る。
ENTER_MAX=22000
THR_OK=3.0
THR_NG=5.5

# run_dir|tag
RUNS=(
  "output/4DGS/neu3d/baseline_neu3d_4000_trial|trial_baseline_4000"
  "output/4DGS/neu3d/baseline_neu3d_7000_trial|trial_baseline_7000"
  "output/4DGS/neu3d/warmstart_neu3d_4000_trial|trial_warmstart_4000"
  "output/4DGS/neu3d/warmstart_neu3d_7000_trial|trial_warmstart_7000"
)

mkdir -p "$LOG_DIR" eval/results

echo "========================================="
echo " trial 残り4本 描画 + 競合実測  $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================="

current_frame () {
    local d
    for d in $(ls -d ${TRAIN_ROOT}/frame_* 2>/dev/null | sort); do
        if [ -f "$d/config.yaml" ] && [ ! -f "$d/checkpoints/latest.pt" ]; then
            echo "$d"; return 0
        fi
    done
    return 1
}
iter_of () {
    tail -1 "$1/train_log.jsonl" 2>/dev/null \
        | python3 -c 'import json,sys
try: print(json.load(sys.stdin)["iteration"])
except Exception: print(-1)' 2>/dev/null || echo -1
}

echo ""
echo "--- 平坦域 (${FLAT_ITER} <= iter <= ${ENTER_MAX}) のフレームを待機 ---"
while :; do
    FRAME=$(current_frame) || { echo "  [中止] 学習中のフレームがありません"; exit 1; }
    IT=$(iter_of "$FRAME")
    if [ "$IT" -ge "$FLAT_ITER" ] 2>/dev/null && [ "$IT" -le "$ENTER_MAX" ] 2>/dev/null; then
        echo "  $(date '+%H:%M:%S')  $(basename "$FRAME") iter ${IT} → 描画開始"
        break
    fi
    echo "  $(date '+%H:%M:%S')  $(basename "$FRAME") iter ${IT} (待機中) ..."
    sleep 20
done
MEASURE_FRAME="$FRAME"

nvidia-smi --query-gpu=timestamp,memory.used,memory.free,utilization.gpu \
    --format=csv -l 5 > "${LOG_DIR}/trial_rest_vram.log" 2>&1 &
VRAM_PID=$!
trap 'kill ${VRAM_PID} 2>/dev/null || true' EXIT

# --- 描画 4 本を連続実行 ---------------------------------------------------
BATCH_START=$(date +%s.%N)
WINDOWS=""
FAILED=""
for entry in "${RUNS[@]}"; do
    RUN_DIR="${entry%%|*}"
    TAG="${entry##*|}"
    OUT_DIR="output/4DGS/neu3d/renders/${TAG}"
    mkdir -p "$OUT_DIR"
    echo ""
    echo "--- [${TAG}] 描画 開始 $(date '+%H:%M:%S') ---"
    T0=$(date +%s.%N)
    python eval/render_4d.py \
        --run_dir "$RUN_DIR" --data_dir "$DATA_DIR" --out_dir "$OUT_DIR" \
        --split test --render-backend cuda \
        2>&1 | tee "${LOG_DIR}/${TAG}_render.log"
    RC=${PIPESTATUS[0]}
    T1=$(date +%s.%N)
    if [ "$RC" -ne 0 ]; then
        echo "  [失敗] ${TAG} 描画 rc=${RC}"
        FAILED="${FAILED} ${TAG}"
        continue
    fi
    echo "  PNG $(find "$OUT_DIR/renders" -name '*.png' | wc -l) 枚  ($(echo "$T1 $T0" | awk '{printf "%.0f", $1-$2}')s)"
    WINDOWS="${WINDOWS}${TAG},${T0},${T1}\n"
done
BATCH_END=$(date +%s.%N)
kill ${VRAM_PID} 2>/dev/null || true

# --- 競合判定 --------------------------------------------------------------
echo ""
echo "--- 競合判定 ---"
MEASURE_FRAME="$MEASURE_FRAME" BATCH_START="$BATCH_START" BATCH_END="$BATCH_END" \
WINDOWS="$WINDOWS" FLAT_ITER="$FLAT_ITER" THR_OK="$THR_OK" THR_NG="$THR_NG" \
python3 <<'PYEOF'
import json, os, statistics, datetime

frame = os.environ["MEASURE_FRAME"]
bs, be = float(os.environ["BATCH_START"]), float(os.environ["BATCH_END"])
flat = int(os.environ["FLAT_ITER"])
ok_thr, ng_thr = float(os.environ["THR_OK"]), float(os.environ["THR_NG"])
wins = [l.split(",") for l in os.environ["WINDOWS"].replace("\\n", "\n").split("\n") if l.strip()]

start = os.stat(os.path.join(frame, "config.yaml")).st_mtime
rows = []
with open(os.path.join(frame, "train_log.jsonl")) as f:
    for line in f:
        try:
            d = json.loads(line)
        except Exception:
            continue
        if d.get("iteration", 0) >= flat:
            rows.append(d)

fmt = lambda t: datetime.datetime.fromtimestamp(t).strftime("%H:%M:%S")
print(f"  計測フレーム : {os.path.basename(frame)} (開始 {fmt(start)})")
print(f"  バッチ全体   : {fmt(bs)} 〜 {fmt(be)}  ({be-bs:.0f}s)")
for tag, t0, t1 in wins:
    print(f"    {tag:24s} {fmt(float(t0))} 〜 {fmt(float(t1))}  ({float(t1)-float(t0):.0f}s)")

if len(rows) < 2:
    print("  [判定不能] 平坦域の記録が不足")
    raise SystemExit(0)

pre, during, post = [], [], []
for a, b in zip(rows, rows[1:]):
    ta, tb = start + a["elapsed_seconds"], start + b["elapsed_seconds"]
    delta = b["elapsed_seconds"] - a["elapsed_seconds"]
    if tb <= bs:   pre.append(delta)
    elif ta >= be: post.append(delta)
    else:          during.append((b["iteration"], delta))

def show(label, xs):
    if not xs:
        print(f"  {label}: 記録なし"); return None
    m = statistics.mean(xs)
    s = statistics.pstdev(xs) if len(xs) > 1 else 0.0
    print(f"  {label}: mean={m:.3f}s/100it  sd={s:.3f}  n={len(xs)}")
    return m

print()
m_pre = show("描画前", pre)
if during:
    ds = [d for _, d in during]
    print(f"  描画中: mean={statistics.mean(ds):.3f}s/100it  "
          f"worst={max(ds):.3f}  n={len(ds)}")
    over = [(i, d) for i, d in during if d >= ok_thr]
    if over:
        print("    しきい値超過:")
        for i, d in over:
            print(f"      iter {i:6d}: {d:.3f}s/100it")
    else:
        print("    全区間しきい値以下")
else:
    print("  描画中: 記録なし")
m_post = show("描画後", post)

print()
if not during:
    print("  判定: 測定失敗 — バッチが学習フレームと重なりませんでした")
else:
    worst = max(d for _, d in during)
    mean_d = statistics.mean([d for _, d in during])
    if worst < ok_thr:
        print("  判定: 影響なし")
    elif worst < ng_thr:
        print("  判定: 軽度競合 (許容範囲)")
    else:
        print("  判定: 重度競合")
    if m_pre:
        print(f"  描画前比 {mean_d/m_pre:.2f}x")
PYEOF

# --- 評価 ------------------------------------------------------------------
for entry in "${RUNS[@]}"; do
    TAG="${entry##*|}"
    OUT_DIR="output/4DGS/neu3d/renders/${TAG}"
    case " ${FAILED} " in *" ${TAG} "*) echo "[スキップ] ${TAG}"; continue;; esac
    echo ""
    echo "############ ${TAG} 評価 ############"
    python eval/evaluate.py \
        --dataset neu3d \
        --render_dir "$OUT_DIR/renders" --gt_dir "$OUT_DIR/gt" \
        --output_csv "eval/results/${TAG}.csv" \
        --device cuda --rgba_background black \
        2>&1 | tee "${LOG_DIR}/${TAG}_eval.log"
    python eval/evaluate_per_frame.py \
        --render_dir "$OUT_DIR/renders" --gt_dir "$OUT_DIR/gt" \
        --output_csv "eval/results/${TAG}_per_frame.csv" \
        --dataset neu3d --device cuda --rgba_background black --block 2 \
        2>&1 | tee "${LOG_DIR}/${TAG}_eval_per_frame.log"
done

# --- 5 本まとめ ------------------------------------------------------------
echo ""
echo "========================================="
echo " trial 5 本 まとめ"
echo "========================================="
python3 <<'PYEOF'
import csv, os, statistics
tags = ["trial_baseline_4000", "trial_baseline_5000", "trial_baseline_7000",
        "trial_warmstart_4000", "trial_warmstart_7000"]
print(f"  {'run':24s} {'n':>3s} {'PSNR↑':>9s} {'D-SSIM↓':>9s} {'LPIPS↓':>9s}")
print("  " + "-" * 62)
for t in tags:
    p = f"eval/results/{t}.csv"
    if not os.path.isfile(p):
        print(f"  {t:24s}  (未実行)"); continue
    rows = [r for r in csv.DictReader(open(p))
            if r["filename"].lower().endswith(".png")]   # *** AVERAGE *** 行を除く
    if not rows:
        print(f"  {t:24s}  (空)"); continue
    f = lambda k: statistics.mean(float(r[k]) for r in rows)
    print(f"  {t:24s} {len(rows):3d} {f('psnr'):9.4f} {f('d_ssim'):9.4f} {f('lpips'):9.4f}")
PYEOF

echo ""
echo " 完了 $(date '+%Y-%m-%d %H:%M:%S')"
[ -n "${FAILED}" ] && echo " 失敗したラン:${FAILED}"
echo "========================================="
