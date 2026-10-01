#!/usr/bin/env python3
"""Merge per-country V6 prediction files into output/ and check row coverage against test_source1."""
import gzip
import os
import sys

COUNTRIES = ("India", "US", "France")
SRC = "work/out_final"   # per-country predictions written by `src/v6_pipeline.py predict --country ...`
DST = "output"


def rows(path):
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt", encoding="utf-8") as f:
        header = f.readline()
        return header, [line for line in f]


def main():
    s1 = []
    with open("dataset/test/test_source1.tsv", encoding="utf-8") as f:
        f.readline()
        for line in f:
            s1.append(line.split("\t", 1)[0].strip())
    os.makedirs(DST, exist_ok=True)
    for kind, header_name in (("matching_results", "matched_entity_ids"), ("candidate_pairs", "candidate_entity_ids")):
        merged = {}
        for c in COUNTRIES:
            path = next((p for p in (f"{SRC}/{kind}_{c}.tsv.gz", f"{SRC}/{kind}_{c}.tsv") if os.path.exists(p)), None)
            if path is None:
                sys.exit(f"missing {kind} for {c}")
            _, lines = rows(path)
            for line in lines:
                sid, _, ids = line.rstrip("\n").partition("\t")
                merged[sid] = ids
            print(f"{kind} {c}: {len(lines):,} rows")
        missing = [s for s in s1 if s not in merged]
        print(f"{kind}: {len(merged):,} merged rows | test S1 {len(s1):,} | missing {len(missing):,}")
        with open(f"{DST}/{kind}.tsv", "w", encoding="utf-8", newline="\n") as f:
            f.write(f"source1_entity_id\t{header_name}\n")
            for sid in s1:
                f.write(f"{sid}\t{merged.get(sid, '')}\n")
    print("merged into output/")


if __name__ == "__main__":
    main()
