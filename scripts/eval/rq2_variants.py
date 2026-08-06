"""
rq2_variants.py — TN2: P1 × 6 biến thể  (trả lời RQ2)
========================================================
Cố định pipeline = P1 (đưa thẳng vào LLM, không dịch).
Thay đổi biến thể ngôn ngữ: vi_full, vi_nodiacritic, mt_en, bilingual, mixed, teencode.
→ Bảng 2: ASR / TAF / flip_rate theo variant.

Chạy:
  python scripts/eval/rq2_variants.py --split test --model qwen2.5-3b
  python scripts/eval/rq2_variants.py --split dev  --model deepseek --limit 20

Output:
  results/rq2_variants/{model}/run_{split}_{timestamp}.jsonl
  results/rq2_variants/{model}/metrics_{split}_{timestamp}.json
"""

import os, sys, json, time, argparse
from datetime import datetime

# Thêm thư mục gốc vào path để import shared
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(BASE_DIR, "scripts", "eval"))

from shared.data    import load_jsonl, filter_records
from shared.llm     import MODELS
from shared.agent   import run_agent, check_success
from shared.metrics import compute_metrics, print_metrics


def main():
    parser = argparse.ArgumentParser(description="TN2 — RQ2: 6 biến thể × P1")
    parser.add_argument("--split",   choices=["dev", "test"], default="dev")
    parser.add_argument("--model",   choices=list(MODELS), default="qwen2.5-3b")
    parser.add_argument("--variant", default=None, help="Lọc 1 variant (tuỳ chọn)")
    parser.add_argument("--task",    default=None, help="Lọc 1 task_type (tuỳ chọn)")
    parser.add_argument("--limit",   type=int, default=None, help="Giới hạn số record")
    args = parser.parse_args()

    data_path = os.path.join(BASE_DIR, "data", "final", f"{args.split}.jsonl")
    records   = load_jsonl(data_path)
    records   = filter_records(records, variant=args.variant, task=args.task, limit=args.limit)

    ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(BASE_DIR, "results", "rq2_variants", args.model)
    os.makedirs(out_dir, exist_ok=True)
    raw_path = os.path.join(out_dir, f"run_{args.split}_{ts}.jsonl")
    met_path = os.path.join(out_dir, f"metrics_{args.split}_{ts}.json")

    print(f"=== ViTransInject — TN2 / RQ2 (6 Biến thể) ===")
    print(f"Split: {args.split} | Model: {args.model} | Records: {len(records)}")
    if args.variant: print(f"Variant filter: {args.variant}")
    if args.task:    print(f"Task filter: {args.task}")
    print()

    raw_results = []
    total = len(records)

    with open(raw_path, "w", encoding="utf-8") as fout:
        for i, r in enumerate(records):
            print(f"[{i+1}/{total}] {r['id']} ... ", end="", flush=True)
            try:
                response = run_agent(r, args.model)
                sc  = r.get("success_condition")
                hit = check_success(response, sc) if sc else False
                print("MẮC BẪY" if hit else "an toàn")
            except Exception as e:
                response = f"ERROR: {e}"
                hit = False
                print(f"ERROR: {e}")

            result = {
                "id":              r["id"],
                "seed_id":         r["seed_id"],
                "task_type":       r["task_type"],
                "group":           r["group"],
                "variant":         r["variant"],
                "is_benign":       r["is_benign"],
                "attacker_goal":   r.get("attacker_goal"),
                "injection_style": r.get("injection_style"),
                "injection_position": r.get("injection_position"),
                "hit":             hit,
                "response":        response,
            }
            fout.write(json.dumps(result, ensure_ascii=False) + "\n")
            fout.flush()
            raw_results.append(result)
            time.sleep(0.2)

    print(f"\n=== Metrics ===\n")
    metrics = compute_metrics(raw_results, rq="rq2")
    print_metrics(metrics, rq="rq2")

    with open(met_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(f"\n→ Raw:     {raw_path}")
    print(f"→ Metrics: {met_path}")


if __name__ == "__main__":
    main()
