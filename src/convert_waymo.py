"""
Waymo Open Motion (v1.2.1 `uncompressed/scenario`) -> ScenarioNet database, WITHOUT TensorFlow.

ScenarioNet's own converter needs tensorflow only to read the TFRecord container; the Scenario
proto definition ships with ScenarioNet. A TFRecord is just  <u64 length><u32 crc><data><u32 crc>,
so it is read here in pure Python and everything else reuses ScenarioNet's converter unchanged.

    python src/convert_waymo.py --raw <dir of tfrecord shards> --out data/waymo_val \
        --first_shards 8 --workers 3

Only shards that finished downloading are used (a sibling `<shard>.ok` marker must exist, and
`.gstmp` partials are ignored). Record CRCs are not verified (the download tool already checks
integrity); a truncated record aborts that shard loudly instead of yielding a partial scenario.
"""
from __future__ import annotations

import argparse
import logging
import os
import struct
import sys
from pathlib import Path


def read_tfrecord(path: str):
    with open(path, "rb") as f:
        while True:
            head = f.read(8)
            if not head:
                return
            if len(head) < 8:
                raise IOError(f"{path}: truncated length header")
            (n,) = struct.unpack("<Q", head)
            f.read(4)
            data = f.read(n)
            if len(data) < n:
                raise IOError(f"{path}: truncated record ({len(data)}/{n} bytes)")
            f.read(4)
            yield data


def preprocess_no_tf(files, worker_index):
    from scenarionet.converter.waymo.utils import SPLIT_KEY
    from scenarionet.converter.waymo.waymo_protos import scenario_pb2

    cap = int(os.environ.get("P03_WAYMO_MAX", "0")) or None
    n_out = 0
    for file in files:
        for data in read_tfrecord(file):
            if cap is not None and n_out >= cap:
                return
            n_out += 1
            sc = scenario_pb2.Scenario()
            sc.ParseFromString(data)
            sc.scenario_id = sc.scenario_id + SPLIT_KEY + file
            yield sc


def finished_shards(raw: Path) -> list[str]:
    out = []
    for p in sorted(raw.glob("*tfrecord-*")):
        if p.suffix in (".ok", ".gstmp") or p.name.endswith("_.gstmp"):
            continue
        if (p.parent / (p.name + ".ok")).exists():
            out.append(str(p))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--first_shards", type=int, default=8)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--max_scenarios", type=int, default=None, help="debug: stop after N (single shard)")
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()

    from scenarionet.converter.utils import write_to_directory
    from scenarionet.converter.waymo.utils import convert_waymo_scenario

    shards = finished_shards(Path(a.raw))[: a.first_shards]
    if len(shards) < a.first_shards:
        print(f"WARNING: only {len(shards)} finished shards available (asked {a.first_shards})")
    if not shards:
        sys.exit("no finished shards")
    if a.max_scenarios:
        os.environ["P03_WAYMO_MAX"] = str(a.max_scenarios)
    pre = preprocess_no_tf
    logging.basicConfig(level=logging.WARNING)
    write_to_directory(convert_func=convert_waymo_scenario, scenarios=shards, output_path=a.out,
                       dataset_version="v1.2", dataset_name="waymo", overwrite=a.overwrite,
                       num_workers=min(a.workers, len(shards)), preprocess=pre)
    print(f"converted {len(shards)} shards -> {a.out}")


if __name__ == "__main__":
    main()
