cd /f/CLAUDE/AI3/paper03-risk-fallback
while true; do
  echo "$(date '+%F %T') cal5_10hz=$(cat results/campaign/av2_cal5_10hz/part_*.jsonl 2>/dev/null | wc -l) test5_10hz=$(cat results/campaign/av2_test5_10hz/part_*.jsonl 2>/dev/null | wc -l)" >> results/final/progress10.log
  sleep 300
done
