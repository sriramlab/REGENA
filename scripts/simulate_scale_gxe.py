#!/usr/bin/env python3
"""Simulate a toy phenotype whose GxE exists only on the observed scale.

On a latent scale the trait is purely additive, z = g + e with h2 = var(g)/var(z),
identical in both environments. The observed trait is y = exp(a * z + shift * E), so
E multiplies y by exp(shift) and every variance component in E = 1 is exp(2 * shift)
times larger: amplification GxE on the observed scale (Box-Cox lambda = 1) that the
log scale (lambda = 0) removes exactly.

Usage: python3 simulate_scale_gxe.py <plink prefix> <env file> <out.pheno>
       [--h2 0.5] [--a 0.5] [--shift 0.5] [--seed 1]
Needs numpy only; reads the PLINK .bed directly.
"""
import argparse

import numpy as np


def read_bed(prefix):
    n = sum(1 for _ in open(prefix + ".fam"))
    m = sum(1 for _ in open(prefix + ".bim"))
    raw = np.fromfile(prefix + ".bed", dtype=np.uint8)
    if raw[:3].tolist() != [0x6C, 0x1B, 0x01]:
        raise SystemExit("not a SNP-major PLINK .bed file")
    bytes_per_snp = (n + 3) // 4
    packed = raw[3:].reshape(m, bytes_per_snp)
    codes = np.unpackbits(packed[:, :, None], axis=2, bitorder="little")
    codes = codes.reshape(m, bytes_per_snp, 8)
    # 2-bit genotype codes, low bit first: 00 hom A1, 01 missing, 10 het, 11 hom A2
    lo, hi = codes[:, :, 0::2], codes[:, :, 1::2]
    geno = (lo + hi).astype(np.float64).reshape(m, -1)[:, :n]
    missing = ((lo == 1) & (hi == 0)).reshape(m, -1)[:, :n]
    geno[missing] = np.nan
    return geno.T  # n x m


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("prefix")
    ap.add_argument("env")
    ap.add_argument("out")
    ap.add_argument("--h2", type=float, default=0.5)
    ap.add_argument("--a", type=float, default=0.5)
    ap.add_argument("--shift", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    X = read_bed(args.prefix)
    mu = np.nanmean(X, axis=0)
    sd = np.nanstd(X, axis=0)
    keep = sd > 0
    X = X[:, keep]
    X = np.where(np.isnan(X), mu[keep], X)
    X = (X - mu[keep]) / sd[keep]
    n, m = X.shape

    beta = rng.normal(0.0, np.sqrt(args.h2 / m), m)
    g = X @ beta
    e = rng.normal(0.0, np.sqrt(1.0 - args.h2), n)
    z = g + e
    z = (z - z.mean()) / z.std()

    rows = [line.split() for line in open(args.env)]
    header, rows = rows[0], rows[1:]
    E = np.array([float(r[2]) for r in rows])
    if len(rows) != n:
        raise SystemExit("environment file and .fam have different numbers of rows")

    y = np.exp(args.a * z + args.shift * E)
    with open(args.out, "w") as fh:
        fh.write("FID IID pheno\n")
        for r, v in zip(rows, y):
            fh.write(f"{r[0]} {r[1]} {v:.6f}\n")


if __name__ == "__main__":
    main()
