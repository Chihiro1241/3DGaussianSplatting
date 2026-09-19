#!/usr/bin/env bash
# eval/run_full_neu3d_100.sh
# warmstart_neu3d_full と baseline_neu3d_full (どちらも 2000 iter / frames 1-100)
# を描画・評価し、warm-start の優劣を 100 フレームで判断する。
# baseline_30k の学習と並行して流す。
#
#   setsid nohup bash eval/run_full_neu3d_100.sh > logs/full_100.log 2>&1 &
#
# 競合判定は絶対しきい値をやめ、フレームごとの相対比で見る。
# 平坦域 (iter>=16000) のベースラインはフレーム間で 2.649〜2.947 s/100it と
# 振れるため、固定しきい値 3.0 では frame 固有の速さを競合と誤判定する。

set -uo pipefail

REPO=/home/chihiro-tanaka/Git/3DGaussianSplatting
cd "$REPO" || exit 1

DATA_DIR=data/neu3d/coffee_martini/converted_4d
LOG_DIR=eval/logs
TRAIN_ROOT=output/4DGS/neu3d/baseline_30k
FLAT_ITER=16000
END_FRAME=100

RUNS=(
  "output/4DGS/neu3d/baseline_neu3d_full|full_baseline_2000"
  "output/4DGS/neu3d/warmstart_neu3d_full|full_warmstart_2000"
)

mkdir -p "$LOG_DIR" eval/results

echo "========================================="
echo " full 100 フレーム 描画 + 評価  $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================="

nvidia-smi --query-gpu=timestamp,memory.used,memory.free,utilization.gpu \
    --format=csv -l 5 > "${LOG_DIR}/full_100_vram.log" 2>&1 &
VRAM_PID=$!
trap 'kill ${VRAM_PID} 2>/dev/null || true' EXIT

BATCH_START=$(date +%s.%N)
FAILED=""
for entry in "${RUNS[@]}"; do
    RUN_DIR="${entry%%|*}"; TAG="${entry##*|}"
    OUT_DIR="output/4DGS/neu3d/renders/${TAG}"
    mkdir -p "$OUT_DIR"
    echo ""
    echo "--- [${TAG}] 描画 開始 $(date '+%H:%M:%S') ---"
    python eval/render_4d.py \
        --run_dir "$RUN_DIR" --data_dir "$DATA_DIR" --out_dir "$OUT_DIR" \
        --split test --render-backend cuda --end_frame ${END_FRAME} \
        2>&1 | tee "${LOG_DIR}/${TAG}_render.log"
    RC=${PIPESTATUS[0]}
    if [ "$RC" -ne 0 ]; then
        echo "  [失敗] ${TAG} rc=${RC}"; FAILED="${FAILED} ${TAG}"; continue
    fi
    echo "  PNG $(find "$OUT_DIR/renders" -name '*.png' | wc -l) 枚"
done
BATCH_END=$(date +%s.%N)
kill ${VRAM_PID} 2>/dev/null || true

# --- 競合判定 (フレームごとの相対比) ---------------------------------------
echo ""
echo "--- 競合判定 ---"
TRAIN_ROOT="$TRAIN_ROOT" BATCH_START="$BATCH_START" BATCH_END="$BATCH_END" \
FLAT_ITER="$FLAT_ITER" python3 <<'PYEOF'
import json, os, glob, statistics, datetime

root = os.environ["TRAIN_ROOT"]
bs, be = float(os.environ["BATCH_START"]), float(os.environ["BATCH_END"])
flat = int(os.environ["FLAT_ITER"])
fmt = lambda t: datetime.datetime.fromtimestamp(t).strftime("%H:%M:%S")
print(f"  バッチ: {fmt(bs)} 〜 {fmt(be)}  ({be-bs:.0f}s)")

any_during = False
for fd in sorted(glob.glob(os.path.join(root, "frame_*"))):
    cfg = os.path.join(fd, "config.yaml")
    log = os.path.join(fd, "train_log.jsonl")
    if not (os.path.isfile(cfg) and os.path.isfile(log)):
        continue
    start = os.stat(cfg).st_mtime
    rows = []
    for line in open(log):
        try:
            d = json.loads(line)
        except Exception:
            continue
        if d.get("iteration", 0) >= flat:
            rows.append(d)
    if len(rows) < 2:
        continue
    # このフレームの平坦域がバッチ期間と重なるか
    if start + rows[-1]["elapsed_seconds"] < bs or start + rows[0]["elapsed_seconds"] > be:
        continue
    base, during = [], []
    for a, b in zip(rows, rows[1:]):
        ta, tb = start + a["elapsed_seconds"], start + b["elapsed_seconds"]
        delta = b["elapsed_seconds"] - a["elapsed_seconds"]
        (during if (tb > bs and ta < be) else base).append(delta)
    if not during:
        continue
    any_during = True
    md = statistics.mean(during)
    name = os.path.basename(fd)
    if base:
        mb = statistics.mean(base)
        ratio = md / mb
        verdict = ("影響なし" if ratio < 1.15 else
                   "軽度競合" if ratio < 1.5 else "重度競合")
        print(f"  {name}: 非描画中 {mb:.3f} (n={len(base)}) / "
              f"描画中 {md:.3f} (n={len(during)}) s/100it  "
              f"-> {ratio:.2f}x  {verdict}")
    else:
        print(f"  {name}: 描画中 {md:.3f} s/100it (n={len(during)})  "
              f"比較対象なし(平坦域が全部描画中)")
if not any_during:
    print("  [判定不能] バッチ期間に重なる平坦域がありませんでした")
PYEOF

# --- 評価 ------------------------------------------------------------------
for entry in "${RUNS[@]}"; do
    TAG="${entry##*|}"; OUT_DIR="output/4DGS/neu3d/renders/${TAG}"
    case " ${FAILED} " in *" ${TAG} "*) echo "[スキップ] ${TAG}"; continue;; esac
    echo ""
    echo "############ ${TAG} 評価 ############"
    python eval/evaluate.py --dataset neu3d \
        --render_dir "$OUT_DIR/renders" --gt_dir "$OUT_DIR/gt" \
        --output_csv "eval/results/${TAG}.csv" \
        --device cuda --rgba_background black \
        2>&1 | tee "${LOG_DIR}/${TAG}_eval.log"
    python eval/evaluate_per_frame.py \
        --render_dir "$OUT_DIR/renders" --gt_dir "$OUT_DIR/gt" \
        --output_csv "eval/results/${TAG}_per_frame.csv" \
        --dataset neu3d --device cuda --rgba_background black --block 10 \
        2>&1 | tee "${LOG_DIR}/${TAG}_eval_per_frame.log"
done

# --- 突き合わせ ------------------------------------------------------------
echo ""
echo "========================================="
echo " baseline vs warm-start (2000 iter, 100 frames)"
echo "========================================="
python eval/compare_runs.py \
    --a eval/results/full_baseline_2000_per_frame.csv  --a_label "baseline 2000" \
    --b eval/results/full_warmstart_2000_per_frame.csv --b_label "warm-start 2000" \
    --block 10 2>&1 || echo "(compare_runs.py が失敗しました)"

echo ""
echo " 完了 $(date '+%Y-%m-%d %H:%M:%S')"
[ -n "${FAILED}" ] && echo " 失敗:${FAILED}"
echo "========================================="
