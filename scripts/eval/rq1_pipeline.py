"""
rq1_pipeline.py — TN1: vi_full × P1–P4  (trả lời RQ1)
========================================================
Cố định biến thể = vi_full (văn bản tiếng Việt gốc).
Thay đổi pipeline xử lý:
  P1 — đưa thẳng vào LLM (không dịch)       → data/final/{split}.jsonl (vi_full)
  P2 — doc + instr đều dịch sang EN          → data/rq1_pipelines/p2_full_en/
  P3 — chỉ instr dịch sang EN, doc giữ VI   → data/rq1_pipelines/p3_instr_en/
  P4 — doc song ngữ VI+EN, instr giữ VI     → data/rq1_pipelines/p4_bilingual/

Data P2/P3/P4 được tạo trước bằng Google Translate (scripts/gen/gen_rq1_pipelines.py).
Không dịch tại runtime → không circular bias, nhanh hơn, reproducible.

→ Bảng 1: ASR / TAF theo pipeline (RQ1).

Chuẩn bị data (chạy 1 lần):
  python scripts/gen/gen_rq1_pipelines.py

Chạy thực nghiệm:
  CUDA_MPS_PIPE_DIRECTORY=/tmp/fake_mps python scripts/eval/rq1_pipeline.py --split test --model qwen2.5-3b --pipeline P1
  CUDA_MPS_PIPE_DIRECTORY=/tmp/fake_mps python scripts/eval/rq1_pipeline.py --split test --model qwen2.5-3b --pipeline P2
  CUDA_MPS_PIPE_DIRECTORY=/tmp/fake_mps python scripts/eval/rq1_pipeline.py --split test --model qwen2.5-3b --pipeline P3
  CUDA_MPS_PIPE_DIRECTORY=/tmp/fake_mps python scripts/eval/rq1_pipeline.py --split test --model qwen2.5-3b --pipeline P4

Output:
  results/rq1_pipeline/{model}/{pipeline}/run_{split}_{timestamp}.jsonl
  results/rq1_pipeline/{model}/{pipeline}/metrics_{split}_{timestamp}.json
"""

import os, sys, json, time, argparse, glob
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(BASE_DIR, "scripts", "eval"))

from shared.data    import load_jsonl, filter_records
from shared.llm     import MODELS
from shared.agent   import run_agent, check_success
from shared.metrics import compute_metrics, print_metrics

# Map pipeline → thư mục data đã tạo sẵn
PIPELINE_DATA = {
    "P1": os.path.join(BASE_DIR, "data", "final"),           # dùng final/, lọc vi_full
    "P2": os.path.join(BASE_DIR, "data", "rq1_pipelines", "p2_full_en"),
    "P3": os.path.join(BASE_DIR, "data", "rq1_pipelines", "p3_instr_en"),
    "P4": os.path.join(BASE_DIR, "data", "rq1_pipelines", "p4_bilingual"),
}


def load_p1_asr(model_name: str, split: str) -> float | None:
    """Đọc ASR của P1 từ file metrics mới nhất để tính TAF cho P2/P3/P4."""
    p1_dir = os.path.join(BASE_DIR, "results", "rq1_pipeline", model_name, "P1")
    pattern = os.path.join(p1_dir, f"metrics_{split}_*.json")
    files = sorted(glob.glob(pattern))
    if not files:
        return None
    with open(files[-1], encoding="utf-8") as f:
        data = json.load(f)
    return data.get("ASR_overall")


def main():
    parser = argparse.ArgumentParser(description="TN1 — RQ1: vi_full × P1–P4")
    parser.add_argument("--split",    choices=["dev", "test"], default="dev")
    parser.add_argument("--model",    choices=list(MODELS), default="qwen2.5-3b")
    parser.add_argument("--pipeline", choices=["P1", "P2", "P3", "P4"], required=True)
    parser.add_argument("--task",     default=None, help="Lọc 1 task_type (tuỳ chọn)")
    parser.add_argument("--limit",    type=int, default=None, help="Giới hạn số record")
    args = parser.parse_args()

    # Load data từ đúng thư mục theo pipeline
    data_dir  = PIPELINE_DATA[args.pipeline]
    data_path = os.path.join(data_dir, f"{args.split}.jsonl")

    if not os.path.exists(data_path):
        if args.pipeline in ("P2", "P3", "P4"):
            print(f"[!] Data cho {args.pipeline} chưa tồn tại: {data_path}")
            print(f"    Chạy trước: python scripts/gen/gen_rq1_pipelines.py --split {args.split}")
            return
        else:
            print(f"[!] File không tồn tại: {data_path}")
            return

    all_records = load_jsonl(data_path)

    # P1: lọc vi_full từ final/; P2/P3/P4: đã chỉ chứa vi_full, nhưng filter task/limit vẫn cần
    if args.pipeline == "P1":
        records = filter_records(all_records, variant="vi_full", task=args.task, limit=args.limit)
    else:
        records = filter_records(all_records, task=args.task, limit=args.limit)

    ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(BASE_DIR, "results", "rq1_pipeline", args.model, args.pipeline)
    os.makedirs(out_dir, exist_ok=True)
    raw_path = os.path.join(out_dir, f"run_{args.split}_{ts}.jsonl")
    met_path = os.path.join(out_dir, f"metrics_{args.split}_{ts}.json")

    print(f"=== ViTransInject — TN1 / RQ1 (4 Pipeline) ===")
    print(f"Split: {args.split} | Model: {args.model} | Pipeline: {args.pipeline} | Records: {len(records)}")
    print(f"Data:  {data_path}")
    if args.task: print(f"Task filter: {args.task}")
    print()

    raw_results = []
    total = len(records)

    with open(raw_path, "w", encoding="utf-8") as fout:
        for i, r in enumerate(records):
            print(f"[{i+1}/{total}] {r['id']} ... ", end="", flush=True)
            try:
                # P2/P3/P4: doc/instr đã được dịch sẵn trong data, truyền thẳng vào agent
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
                "variant":         r["variant"],   # luôn vi_full
                "pipeline":        args.pipeline,
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
    metrics = compute_metrics(raw_results, rq="rq1")

    # TAF so P1 nếu đang chạy P2/P3/P4
    if args.pipeline != "P1":
        p1_asr = load_p1_asr(args.model, args.split)
        if p1_asr is not None:
            cur_asr = metrics["ASR_overall"]
            taf = round(cur_asr / p1_asr, 4) if p1_asr > 0 else None
            metrics["TAF_vs_P1"] = taf
            print(f"TAF vs P1 ({p1_asr:.1%}): {taf:.2f}" if taf else "TAF vs P1: N/A (P1 ASR=0)")
        else:
            metrics["TAF_vs_P1"] = None
            print("[!] Chưa có kết quả P1 → TAF_vs_P1 = null. Chạy P1 trước để có TAF.")

    print_metrics(metrics, rq="rq1")

    with open(met_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(f"\n→ Raw:     {raw_path}")
    print(f"→ Metrics: {met_path}")


if __name__ == "__main__":
    main()
