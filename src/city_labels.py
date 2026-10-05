"""City label for every stored AV2 scenario (Amendment 9, A1). Output: results/final/av2_city_labels.json
{db: {seed: city}} built from the ScenarioNet file list (position = campaign seed) and the raw AV2 parquet."""
import json, re, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import pandas as pd
import simenv
from scenarionet.common_utils import read_dataset_summary

RAW = Path(r"F:\CLAUDE\AI1\paper01-coverage-transfer\data\argoverse2\val")
out, cache = {}, {}
for db in ("av2_cal", "av2_test"):
    _, lst, _ = read_dataset_summary(simenv.DBS[db])
    labels = {}
    for seed, f in enumerate(lst):
        sid = re.search(r"([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})\.pkl$", f).group(1)
        if sid not in cache:
            pq = RAW / sid / f"scenario_{sid}.parquet"
            cache[sid] = pd.read_parquet(pq, columns=["city"])["city"].iloc[0] if pq.exists() else None
        labels[seed] = cache[sid]
    out[db] = labels
    print(db, len(labels), pd.Series(list(labels.values())).value_counts(dropna=False).to_dict())
Path("results/final/av2_city_labels.json").write_text(json.dumps(out))
