#!/usr/bin/env bash
# eval/baseline_30k_status.sh
# baseline 30k ランの進捗を 1 画面で出す。何度呼んでも副作用は無い。
#   bash eval/baseline_30k_status.sh

cd /home/chihiro-tanaka/Git/3DGaussianSplatting || exit 1

RUN_DIR=output/4DGS/neu3d/baseline_30k
TOTAL=300

echo "=============================================================="
echo " baseline 30k 進捗  $(date '+%Y-%m-%d %H:%M:%S')"
echo "=============================================================="

if pgrep -f "[w]armstart_trainer\.py.*baseline_30k" > /dev/null; then
    echo "  学習プロセス   : 稼働中 (pid $(pgrep -f '[w]armstart_trainer\.py.*baseline_30k' | head -1))"
else
    echo "  学習プロセス   : 停止"
fi
if pgrep -f "[r]un_both_baselines" > /dev/null; then
    echo "  統括スクリプト : 稼働中 (30k -> 後段 -> 7k -> 後段 -> 比較)"
else
    echo "  統括スクリプト : 停止"
fi
if pgrep -f "[w]armstart_trainer\.py.*baseline_7k" > /dev/null; then
    echo "  7k 学習        : 稼働中"
fi
if [ -d output/4DGS/neu3d/baseline_7k ]; then
    echo "  7k 完了フレーム: $(find output/4DGS/neu3d/baseline_7k -maxdepth 3 -name latest.pt 2>/dev/null | wc -l)/${TOTAL}"
fi

DONE=$(find "$RUN_DIR" -maxdepth 3 -name latest.pt 2>/dev/null | wc -l)
echo "  完了フレーム   : ${DONE}/${TOTAL}"

if [ "$DONE" -gt 0 ]; then
    # 完了フレームの平均所要時間から残りを見積もる。
    python - "$RUN_DIR" "$DONE" "$TOTAL" <<'PYEOF'
import json, sys
from pathlib import Path
run_dir, done, total = Path(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
times, counts = [], []
for path in sorted(run_dir.glob("frame_*/training_telemetry.json")):
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        continue
    if data.get("status") == "COMPLETED":
        times.append(data.get("training_wall_time_seconds") or 0.0)
        counts.append(data.get("gaussian_count") or 0)
if times:
    average = sum(times) / len(times)
    remaining = (total - done) * average
    print(f"  平均         : {average:.0f} 秒/frame")
    print(f"  残り見込み   : {remaining/3600:.1f} 時間 ({remaining/86400:.2f} 日)")
if counts:
    print(f"  Gaussian 数  : 最小 {min(counts):,} / 平均 {sum(counts)//len(counts):,} / 最大 {max(counts):,}")
PYEOF

    # 進行中フレームの中身。
    CURRENT=$(ls -d "$RUN_DIR"/frame_* 2>/dev/null | tail -1)
    if [ -n "$CURRENT" ] && [ ! -f "$CURRENT/checkpoints/latest.pt" ]; then
        LAST=$(tail -1 "$CURRENT/train_log.jsonl" 2>/dev/null)
        if [ -n "$LAST" ]; then
            echo "  学習中       : $(basename "$CURRENT")"
            echo "$LAST" | python -c "import json,sys; d=json.load(sys.stdin); print(f'    iteration {d[\"iteration\"]:,}/30,000  psnr {d[\"psnr\"]:.2f}  gaussians {d[\"gaussian_count\"]:,}')" 2>/dev/null
        fi
    fi
fi

echo "  ディスク使用   : $(du -sh "$RUN_DIR" 2>/dev/null | cut -f1)  (空き $(df -h . | tail -1 | awk '{print $4}'))"

# 直近のエラーだけ拾う (長時間ランで静かに落ちていないか)。
if [ -f logs/baseline_30k.log ]; then
    ERRORS=$(grep -cE "Traceback|Error|FAILED|CUDA out of memory" logs/baseline_30k.log 2>/dev/null)
    echo "  ログ中のエラー : ${ERRORS} 件"
    if [ "${ERRORS:-0}" -gt 0 ]; then
        echo "  --- 直近 ---"
        grep -E "Traceback|Error|FAILED|CUDA out of memory" logs/baseline_30k.log | tail -3 | sed 's/^/    /'
    fi
fi
echo "=============================================================="
