#!/usr/bin/env python3
"""Print the WIDE_TABLE block of a REGENA output file as a tab-separated table.

The WIDE_TABLE has one row per fitted phenotype. With -bx (Box-Cox) the rows are the
lambda grid in order, so each row is the model fitted on one scale.

Usage: python3 extract_wide_table.py <regena.out> [> table.tsv]
"""
import sys


def read_wide_table(path):
    rows, inside = [], False
    with open(path) as fh:
        for line in fh:
            line = line.rstrip("\n")
            if line == "WIDE_TABLE":
                inside, rows = True, []
            elif inside and line.startswith("*****"):
                inside = False
            elif inside and line:
                rows.append(line.split("\t"))
    if not rows:
        sys.exit(f"no WIDE_TABLE block in {path} (did the run finish?)")
    return rows


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    for row in read_wide_table(sys.argv[1]):
        print("\t".join(row))
