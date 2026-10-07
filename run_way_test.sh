PY=F:/CLAUDE/AI1/shared/envs/unitraj/Scripts/python.exe
WAY='F:\CLAUDE\AI1\paper01-coverage-transfer\results\ckpts\av2_wayformer\epoch19-minADE0.967.ckpt'
$PY src/campaign.py --db av2_test --n 900 --workers 8 --start 3584 --decide_every 5 --mc 0 --mrm_decel 4.0 --method wayformer --ckpt "$WAY" --out results/campaign/av2_test5_way > results/final/way_test.log 2>&1
