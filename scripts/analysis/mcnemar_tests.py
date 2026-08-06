"""
mcnemar_tests.py — McNemar paired test cho tất cả cặp so sánh
==============================================================
RQ1: P1 vs P2/P3/P4 (ghép cặp theo id, cùng model)
RQ2: vi_full vs từng variant (ghép cặp theo seed_id trong cùng model)
RQ3: none vs keyword/spotlight/guard_v3 (ghép cặp theo id, cùng model)

Dùng statsmodels.stats.contingency_tables.mcnemar (exact=True nếu b+c<25)

Output: results/analysis/mcnemar_results.json + in bảng ra stdout
"""

import json, os, glob, sys
from collections import defaultdict
from itertools import combinations

try:
    from statsmodels.stats.contingency_tables import mcnemar as sm_mcnemar
except ImportError:
    print("Cài statsmodels: pip install statsmodels")
    sys.exit(1)

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODELS = ["llama-3.2-1b", "qwen2.5-1.5b", "gemma-2-2b", "qwen2.5-3b", "sailor2-1b"]


def load_latest_run(pattern):
    files = sorted(glob.glob(pattern))
    if not files:
        return None
    lines = [json.loads(l) for l in open(files[-1], encoding="utf-8") if l.strip()]
    return lines


def mcnemar_test(hits_a, hits_b, ids):
    """hits_a, hits_b: dict {id: bool}. ids: list of common ids."""
    b = sum(1 for i in ids if hits_a[i] and not hits_b[i])   # A=1, B=0
    c = sum(1 for i in ids if not hits_a[i] and hits_b[i])   # A=0, B=1
    n = len(ids)
    if b + c == 0:
        return {"b": 0, "c": 0, "n": n, "p": 1.0, "stat": 0.0, "note": "identical"}
    exact = (b + c) < 25
    table = [[0, b], [c, 0]]   # contingency table for mcnemar
    result = sm_mcnemar([[sum(hits_a[i] and hits_b[i] for i in ids),     b],
                         [c,  sum(not hits_a[i] and not hits_b[i] for i in ids)]],
                        exact=exact, correction=True)
    return {
        "b": b, "c": c, "n": n,
        "p": round(float(result.pvalue), 4),
        "stat": round(float(result.statistic), 4),
        "exact": exact,
        "note": "exact" if exact else "chi2+continuity"
    }


def sig_stars(p):
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    if p < 0.1:   return "."
    return "ns"


all_results = {}

# ─── RQ1: P1 vs P2/P3/P4 ────────────────────────────────────────────────────
print("\n" + "="*70)
print("RQ1 — McNemar: P1 vs pipeline (attack records only)")
print("="*70)
print(f"{'Model':<16} {'Cặp':<8} {'b':>4} {'c':>4} {'n':>5} {'p':>8}  sig")

rq1_results = {}
for model in MODELS:
    rq1_results[model] = {}
    p1_data = load_latest_run(f"{BASE}/results/rq1_pipeline/{model}/P1/run_test_*.jsonl")
    if not p1_data:
        continue
    hits_p1 = {r["id"]: r["hit"] for r in p1_data if r["group"] == "attack"}

    for px in ["P2", "P3", "P4"]:
        px_data = load_latest_run(f"{BASE}/results/rq1_pipeline/{model}/{px}/run_test_*.jsonl")
        if not px_data:
            continue
        hits_px = {r["id"]: r["hit"] for r in px_data if r["group"] == "attack"}
        common = [i for i in hits_p1 if i in hits_px]
        res = mcnemar_test(hits_p1, hits_px, common)
        rq1_results[model][f"P1_vs_{px}"] = res
        stars = sig_stars(res["p"])
        print(f"  {model:<14} {'P1↔'+px:<8} {res['b']:>4} {res['c']:>4} {res['n']:>5} {res['p']:>8.4f}  {stars}")

all_results["RQ1"] = rq1_results

# ─── RQ2: vi_full vs variants ────────────────────────────────────────────────
print("\n" + "="*70)
print("RQ2 — McNemar: vi_full vs variant (attack records only)")
print("="*70)
VARIANTS = ["vi_nodiacritic", "mt_en", "bilingual", "mixed", "teencode"]
print(f"{'Model':<16} {'Variant':<18} {'b':>4} {'c':>4} {'n':>5} {'p':>8}  sig")

rq2_results = {}
for model in MODELS:
    rq2_results[model] = {}
    rq2_data = load_latest_run(f"{BASE}/results/rq2_variants/{model}/run_test_*.jsonl")
    if not rq2_data:
        continue
    attacks = [r for r in rq2_data if r["group"] == "attack"]
    # group by variant
    by_variant = defaultdict(dict)
    for r in attacks:
        # Key by seed_id (strip variant suffix) — dùng seed_id field
        by_variant[r["variant"]][r["seed_id"]] = r["hit"]

    baseline = by_variant.get("vi_full", {})
    for var in VARIANTS:
        variant_hits = by_variant.get(var, {})
        common = [sid for sid in baseline if sid in variant_hits]
        if not common:
            continue
        res = mcnemar_test(baseline, variant_hits, common)
        rq2_results[model][f"vi_full_vs_{var}"] = res
        stars = sig_stars(res["p"])
        print(f"  {model:<14} {var:<18} {res['b']:>4} {res['c']:>4} {res['n']:>5} {res['p']:>8.4f}  {stars}")

all_results["RQ2"] = rq2_results

# ─── RQ3: none vs defenses ───────────────────────────────────────────────────
print("\n" + "="*70)
print("RQ3 — McNemar: none vs defense (attack records only)")
print("="*70)
DEFENSES = ["keyword", "spotlight", "guard_v3"]
print(f"{'Model':<16} {'Defense':<14} {'b':>4} {'c':>4} {'n':>5} {'p':>8}  sig")

rq3_results = {}
for model in MODELS:
    rq3_results[model] = {}
    none_data = load_latest_run(f"{BASE}/results/rq3_guard/{model}/none/run_test_*.jsonl")
    if not none_data:
        continue
    hits_none = {r["id"]: r["hit"] for r in none_data if r["group"] == "attack"}

    for defense in DEFENSES:
        def_data = load_latest_run(f"{BASE}/results/rq3_guard/{model}/{defense}/run_test_*.jsonl")
        if not def_data:
            continue
        hits_def = {r["id"]: r["hit"] for r in def_data if r["group"] == "attack"}
        common = [i for i in hits_none if i in hits_def]
        res = mcnemar_test(hits_none, hits_def, common)
        rq3_results[model][f"none_vs_{defense}"] = res
        stars = sig_stars(res["p"])
        print(f"  {model:<14} {defense:<14} {res['b']:>4} {res['c']:>4} {res['n']:>5} {res['p']:>8.4f}  {stars}")

all_results["RQ3"] = rq3_results

# ─── Save ────────────────────────────────────────────────────────────────────
out_dir = os.path.join(BASE, "results", "analysis")
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "mcnemar_results.json")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(all_results, f, ensure_ascii=False, indent=2)

print(f"\n→ Saved: {out_path}")
print("\nLegend: b = A=hit B=miss | c = A=miss B=hit | sig: *** p<0.001  ** p<0.01  * p<0.05  . p<0.1  ns p≥0.1")
