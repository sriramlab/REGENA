#!/usr/bin/env python3
"""Select the reporting scale from a REGENA Box-Cox scan.

REGENA (-bx) fits the G+GxE+NxE model on a grid of Box-Cox scales and writes one row per
lambda to the WIDE_TABLE block of its output. This script turns that scan into one
reported scale per trait x environment pair:

1. Valid region. Only scales with a plausible Box-Cox likelihood are considered. With
   z_i(lambda) = f_lambda(y_i) ~ N(mu, sigma^2), the profile log-likelihood is
       l_p(lambda) = -(n/2) log sigma_hat^2(lambda) + (lambda - 1) sum_i log y_i,
   and a lambda is kept when (l_p(lambda) - max l_p) / n >= -ll_drop, restricted to the
   contiguous block of the grid that contains the maximum. Scales where either
   environment's sigma^2_g is not positive are also dropped.

2. Two GxE tests at every kept scale (two-sided Wald):
       amplification            H0: sigma^2_g0 = sigma^2_g1
       imperfect correlation    H0: rho_g = 1
   S = {lambda : neither test rejects at alpha}.

3. If S is non-empty the pair is SCALE-DEPENDENT: some scale removes the GxE. The reporting
   scale is the member of S that maximises (1 - w) sigma^2_g0 + w sigma^2_g1, with w the
   mean of the environment column.

4. If S is empty the pair is SCALE-INDEPENDENT: no Box-Cox scale removes the GxE. Its
   optimal scale lambda* is the projection of lambda = 1 onto
       A = ({p_amp > alpha_set} U {p_rho > alpha_set}) n {min_e z(sigma^2_g,e) >= min_z},
   i.e. the member of A closest to 1, and lambda* = 1 (the valid grid point nearest to 1)
   when A is empty. lambda* moves off the observed scale only when a scale removes one GxE
   component, and by the least amount that does. The component still present at lambda*
   is reported as remaining_gxe.
   If one environment's sigma^2_g is not distinguishable from zero at the observed scale
   (z < min_z), S can be empty for a reason unrelated to scale: weak_sigma2_g_at_observed
   flags such pairs.

Environment labels: a one-column 0/1 environment file E is expanded to [E, 1 - E], so
env0 in REGENA's output is the E = 1 stratum and env1 is the E = 0 stratum.

Usage:
    python3 regena_select_scale.py --out trait.out --pheno trait.pheno --env env.bin.env
        [--alpha 0.05] [--alpha-set 0.05] [--ll-drop 0.5] [--min-z 2]
        [--trajectory scan.tsv]

--pheno must be the file REGENA was run on (untransformed; column 3 is the phenotype,
-9 is missing). The result is printed as a two-column table (field, value).
"""
import argparse
import io
import sys
from math import erfc, sqrt

import numpy as np
import pandas as pd

S0, S1, RHO = "sigma2_g_env0_bin0", "sigma2_g_env1_bin0", "rho_bin0_env0-1"
H0, H1 = "h2_g_env0_bin0", "h2_g_env1_bin0"
MISSING = (-9,)


# ----------------------------------------------------------------------------
# Inputs
# ----------------------------------------------------------------------------
def read_wide_table(path):
    with open(path) as fh:
        lines = fh.readlines()
    start = next((i for i, ln in enumerate(lines) if ln.strip().startswith("WIDE_TABLE")), None)
    if start is None:
        sys.exit(f"no WIDE_TABLE block in {path} (did the run finish?)")
    df = pd.read_csv(io.StringIO("".join(lines[start + 1:])), comment="*", sep="\t", dtype=float)
    if "lambda" not in df.columns:
        sys.exit(f"{path} has no lambda column: run REGENA with -bx to scan Box-Cox scales")
    return df[np.isfinite(df["lambda"])].reset_index(drop=True)


def read_column(path):
    return pd.read_csv(path, sep=r"\s+").iloc[:, 2].to_numpy(float)


def positive_observed(y):
    y = np.asarray(y, float)
    ok = np.isfinite(y) & ~np.isin(y, np.asarray(MISSING, float)) & (y > 0)
    return y[ok]


# ----------------------------------------------------------------------------
# Valid region: Box-Cox profile log-likelihood window
# ----------------------------------------------------------------------------
def boxcox_profile_ll(y_pos, lam, clip_t=700.0, min_var=1e-12):
    """-(n/2) log var(z_lambda) + (lambda - 1) sum log y, up to an additive constant."""
    n = y_pos.size
    if n < 2:
        return -np.inf
    logy = np.log(y_pos)
    if np.isclose(float(lam), 0.0):
        z = logy
    else:
        t = float(lam) * logy
        if np.nanmax(np.abs(t)) > clip_t:
            return -np.inf
        z = np.expm1(t) / float(lam)
    if not np.all(np.isfinite(z)):
        return -np.inf
    v = np.var(z, ddof=0)
    if (not np.isfinite(v)) or v <= min_var:
        return -np.inf
    return -(n / 2.0) * np.log(v) + (float(lam) - 1.0) * np.sum(logy)


def ll_window(lams, y, ll_drop):
    """Lambdas within ll_drop (per sample) of the maximum, contiguous around the MLE."""
    y_pos = positive_observed(y)
    n = y_pos.size
    if n < 2:
        sys.exit("phenotype has fewer than 2 positive, non-missing values")
    lam_u = np.unique(lams)
    ll = np.array([boxcox_profile_ll(y_pos, l) for l in lam_u], float)
    if not np.isfinite(ll).any():
        sys.exit("profile log-likelihood is non-finite at every lambda")
    j = int(np.nanargmax(np.where(np.isfinite(ll), ll, -np.inf)))
    keep = np.isfinite(ll) & ((ll - ll[j]) / n >= -ll_drop)
    lo = hi = j
    while lo - 1 >= 0 and keep[lo - 1]:
        lo -= 1
    while hi + 1 < keep.size and keep[hi + 1]:
        hi += 1
    rel = dict(zip(lam_u, (ll - ll[j]) / n))
    return set(lam_u[lo:hi + 1]), float(lam_u[j]), rel


# ----------------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------------
def p_two_sided(z):
    return erfc(abs(float(z)) / sqrt(2.0))


def p_rho_eq_1(rho, se):
    return p_two_sided((float(rho) - 1.0) / max(float(se), 1e-300))


def p_sigma_eq(s0, se0, s1, se1):
    v = max(float(se0), 0.0) ** 2 + max(float(se1), 0.0) ** 2
    return p_two_sided((float(s0) - float(s1)) / max(sqrt(v), 1e-300))


def scan_table(wide, y, ll_drop):
    kept, lam_mle, rel_ll = ll_window(wide["lambda"].to_numpy(float), y, ll_drop)
    d = wide.copy()
    d["rel_profile_ll"] = [rel_ll[l] for l in d["lambda"]]
    d["in_valid_region"] = d["lambda"].isin(kept)
    d["sigma2_g_positive"] = (np.isfinite(d[S0]) & np.isfinite(d[S1]) & np.isfinite(d[RHO])
                              & (d[S0] > 0) & (d[S1] > 0))
    d["p_amp"] = [p_sigma_eq(a, ae, b, be) for a, ae, b, be
                  in zip(d[S0], d[S0 + "_se"], d[S1], d[S1 + "_se"])]
    d["p_rho"] = [p_rho_eq_1(r, rs) for r, rs in zip(d[RHO], d[RHO + "_se"])]
    d["min_z_sigma2_g"] = np.minimum(d[S0] / d[S0 + "_se"], d[S1] / d[S1 + "_se"])
    return d.sort_values("lambda").reset_index(drop=True), lam_mle


# ----------------------------------------------------------------------------
# Scale selection
# ----------------------------------------------------------------------------
def nearest_to_1(lams, ok):
    if not ok.any():
        return None
    cand = np.where(ok)[0]
    return int(cand[np.abs(lams[cand] - 1.0).argmin()])


def select_scale(d, w, alpha, alpha_set, min_z):
    v = d[d.in_valid_region & d.sigma2_g_positive].reset_index(drop=True)
    if v.empty:
        sys.exit("no scale in the valid region has positive sigma^2_g in both environments")
    lams = v["lambda"].to_numpy(float)
    score = (1.0 - w) * v[S0].to_numpy(float) + w * v[S1].to_numpy(float)
    in_S = (v.p_amp > alpha).to_numpy() & (v.p_rho > alpha).to_numpy()
    k1 = int(np.abs(lams - 1.0).argmin())   # the observed scale (nearest grid point)
    res = dict(S_size=int(in_S.sum()), remaining_gxe="NA")

    if in_S.any():
        cand = np.where(in_S)[0]
        i = int(cand[score[cand].argmax()])
        res.update(classification="scale-dependent", rule="max_score_in_S")
    else:
        a_ok = (v.p_amp > alpha_set).to_numpy()
        r_ok = (v.p_rho > alpha_set).to_numpy()
        identified = (v.min_z_sigma2_g >= min_z).to_numpy()
        j = nearest_to_1(lams, (a_ok | r_ok) & identified)
        if j is None:
            i, rule, remaining = k1, "lambda1_no_scale_removes_a_component", "amplification+heterogeneity"
        else:
            i = j
            rule = "lambda1_in_set" if j == k1 else "closest_to_1"
            # the component the scale removed is gone; the other one is the real GxE
            remaining = "heterogeneity" if a_ok[j] and not r_ok[j] else (
                "amplification" if r_ok[j] and not a_ok[j] else "none")
        res.update(classification="scale-independent", rule=rule, remaining_gxe=remaining)

    row = v.iloc[i]
    at1 = v.iloc[k1]
    res.update(
        lambda_selected=float(row["lambda"]),
        valid_lambda_min=float(lams.min()), valid_lambda_max=float(lams.max()),
        n_lambda_valid=int(len(v)),
        w_env=w,
        sigma2_g_env0=row[S0], sigma2_g_env0_se=row[S0 + "_se"],
        sigma2_g_env1=row[S1], sigma2_g_env1_se=row[S1 + "_se"],
        rho_g=row[RHO], rho_g_se=row[RHO + "_se"],
        h2_g_env0=row[H0], h2_g_env0_se=row[H0 + "_se"],
        h2_g_env1=row[H1], h2_g_env1_se=row[H1 + "_se"],
        p_amp=row["p_amp"], p_rho=row["p_rho"],
        min_z_sigma2_g=row["min_z_sigma2_g"],
        lambda_observed=float(at1["lambda"]),
        p_amp_at_observed=at1["p_amp"], p_rho_at_observed=at1["p_rho"],
        min_z_sigma2_g_at_observed=at1["min_z_sigma2_g"],
        # S can be empty because one environment has no detectable genetic variance
        # rather than because of scale; check this flag before interpreting such a pair
        weak_sigma2_g_at_observed=bool(at1["min_z_sigma2_g"] < min_z),
    )
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="REGENA output file from a -bx scan")
    ap.add_argument("--pheno", required=True, help="phenotype file REGENA was run on")
    ap.add_argument("--env", required=True, help="binary environment file REGENA was run on")
    ap.add_argument("--alpha", type=float, default=0.05,
                    help="level of both GxE tests that define S (correct for the number of pairs tested)")
    ap.add_argument("--alpha-set", type=float, default=0.05,
                    help="level of the acceptance sets for lambda* of scale-independent pairs")
    ap.add_argument("--ll-drop", type=float, default=0.5,
                    help="per-sample profile log-likelihood drop that bounds the valid region")
    ap.add_argument("--min-z", type=float, default=2.0,
                    help="minimum z(sigma^2_g) in both environments for a scale to be identifiable")
    ap.add_argument("--trajectory", help="also write the per-lambda scan with tests to this TSV")
    args = ap.parse_args()

    wide = read_wide_table(args.out)
    y = read_column(args.pheno)
    E = read_column(args.env)
    E = E[np.isfinite(E) & ~np.isin(E, np.asarray(MISSING, float))]
    w = float(E.mean())

    d, lam_mle = scan_table(wide, y, args.ll_drop)
    res = select_scale(d, w, args.alpha, args.alpha_set, args.min_z)
    res = dict(lambda_mle=lam_mle, **res)

    if args.trajectory:
        d.to_csv(args.trajectory, sep="\t", index=False, na_rep="NA")
    for k, val in res.items():
        print(f"{k}\t{val:.6g}" if isinstance(val, float) else f"{k}\t{val}")


if __name__ == "__main__":
    main()
