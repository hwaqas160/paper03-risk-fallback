cd /f/CLAUDE/AI3/paper03-risk-fallback
while true; do
  echo "$(date '+%F %T') cal_way=$(cat results/campaign/av2_cal5_way/part_*.jsonl 2>/dev/null | wc -l) test_way=$(cat results/campaign/av2_test5_way/part_*.jsonl 2>/dev/null | wc -l) spatial=$([ -f results/final/spatial_groups.json ] && echo done || echo running)" >> results/final/progress.log
  sleep 300
done
