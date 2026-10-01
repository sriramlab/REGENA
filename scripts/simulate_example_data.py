#!/usr/bin/env python3
"""Simulate the synthetic inputs of the REGENA toy example.

Everything is drawn from a seeded random number generator; no real genotype or phenotype
data is used. For N individuals and M SNPs it writes, next to <prefix>:

    <prefix>.bed/.bim/.fam   genotypes: independent SNPs in Hardy-Weinberg equilibrium,
                             minor allele frequency ~ Uniform(0.05, 0.5), no missing calls
    <prefix>.cov             5 covariates: one binary, four standard normal
    <prefix>.env             binary environment, P(E = 1) = 0.4
    single.annot             one annotation holding every SNP
    multi.annot              two annotations, each SNP assigned to one at random

Phenotypes are simulated on these genotypes by simulate_scale_gxe.py.

Usage: python3 simulate_example_data.py <prefix> [--n 5000] [--m 10000] [--seed 1]
Needs numpy only. The legacy RandomState generator is used on purpose: its stream is
frozen across numpy versions, so a given seed always writes the same files.
"""
import argparse
import os

import numpy as np


def write_bed(path, geno):
    """geno: M x N array of A1 allele counts (0/1/2). Writes a SNP-major PLINK .bed."""
    m, n = geno.shape
    # 2-bit codes, low bit first: 2 copies of A1 -> 00, 1 copy -> 10, 0 copies -> 11
    code = np.array([3, 2, 0], dtype=np.uint8)[geno]
    pad = (-n) % 4
    if pad:
        code = np.concatenate([code, np.zeros((m, pad), dtype=np.uint8)], axis=1)
    code = code.reshape(m, -1, 4)
    packed = code[:, :, 0] | (code[:, :, 1] << 2) | (code[:, :, 2] << 4) | (code[:, :, 3] << 6)
    with open(path, "wb") as fh:
        fh.write(bytes([0x6C, 0x1B, 0x01]))
        fh.write(packed.astype(np.uint8).tobytes())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("prefix")
    ap.add_argument("--n", type=int, default=5000, help="number of individuals")
    ap.add_argument("--m", type=int, default=10000, help="number of SNPs")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    n, m = args.n, args.m
    outdir = os.path.dirname(args.prefix) or "."

    rng = np.random.RandomState(args.seed)

    # Genotypes: redraw the (rare) SNPs that come out monomorphic, REGENA needs MAF > 0.
    maf = rng.uniform(0.05, 0.5, m)
    geno = rng.binomial(2, maf[:, None], (m, n)).astype(np.int8)
    mono = np.flatnonzero(geno.min(axis=1) == geno.max(axis=1))
    while mono.size:
        geno[mono] = rng.binomial(2, maf[mono, None], (mono.size, n))
        mono = mono[geno[mono].min(axis=1) == geno[mono].max(axis=1)]
    write_bed(args.prefix + ".bed", geno)

    ids = np.arange(1, n + 1)
    with open(args.prefix + ".bim", "w") as fh:
        for j in range(m):
            fh.write(f"1\tsim{j + 1}\t0\t{j + 1}\tA\tG\n")
    with open(args.prefix + ".fam", "w") as fh:
        for i in ids:
            fh.write(f"{i} {i} 0 0 0 -9\n")

    cov = np.column_stack([rng.binomial(1, 0.5, n), rng.normal(size=(n, 4))])
    with open(args.prefix + ".cov", "w") as fh:
        fh.write("FID IID cov0 cov1 cov2 cov3 cov4\n")
        for i, row in zip(ids, cov):
            fh.write(f"{i} {i} {int(row[0])} " + " ".join(f"{v:.6f}" for v in row[1:]) + "\n")

    env = rng.binomial(1, 0.4, n)
    with open(args.prefix + ".env", "w") as fh:
        fh.write("FID IID env\n")
        for i, e in zip(ids, env):
            fh.write(f"{i} {i} {e}\n")

    with open(os.path.join(outdir, "single.annot"), "w") as fh:
        fh.write("1\n" * m)
    second = rng.binomial(1, 0.5, m)
    with open(os.path.join(outdir, "multi.annot"), "w") as fh:
        for s in second:
            fh.write(f"{1 - s} {s}\n")


if __name__ == "__main__":
    main()
