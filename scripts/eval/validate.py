"""
validate.py — Bước 6: Kiểm tra chất lượng dataset
===================================================
Kiểm tra toàn bộ data trước khi đóng gói:
  1. Schema đúng và đủ trường
  2. Không trùng id
  3. success_condition nhất quán qua 6 biến thể cùng seed_id
  4. Phân bổ cân bằng (task × variant × attacker_goal × position)
  5. Liệt kê needs_human_review còn lại

Chạy:
  cd /home/iec/disk2/TrongTV/Prompt_Injection
  python scripts/validate.py
"""

import json
import os
import collections

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Tất cả file cần validate
DATA_FILES = [
    # vi_full (attack + clean_plain)
    ("data/attack/attack_vi_full.jsonl",               "attack"),
    ("data/attack/clean_plain_vi_full.jsonl",           "clean_plain"),
    # variants
    ("data/variants/attack_vi_nodiacritic.jsonl",       "attack"),
    ("data/variants/clean_plain_vi_nodiacritic.jsonl",  "clean_plain"),
    ("data/variants/attack_mt_en.jsonl",                "attack"),
    ("data/variants/clean_plain_mt_en.jsonl",           "clean_plain"),
    ("data/variants/attack_bilingual.jsonl",            "attack"),
    ("data/variants/clean_plain_bilingual.jsonl",       "clean_plain"),
    ("data/variants/attack_mixed.jsonl",                "attack"),
    ("data/variants/clean_plain_mixed.jsonl",           "clean_plain"),
    ("data/variants/attack_teencode.jsonl",             "attack"),
    ("data/variants/clean_plain_teencode.jsonl",        "clean_plain"),
    # lookalike
    ("data/lookalike/clean_lookalike_vi_full.jsonl",    "clean_lookalike"),
]

REQUIRED_FIELDS = [
    "id", "seed_id", "task_type", "group", "variant", "lang",
    "user_goal", "trusted_user_instruction", "untrusted_document",
    "ground_truth", "injection_span", "injected_instruction",
    "attacker_goal", "injection_style", "injection_position",
    "target_tool", "success_condition", "is_benign", "needs_human_review",
]

VALID_TASK_TYPES  = {"email", "rag_qa", "cskh", "summarization", "form_extraction"}
VALID_GROUPS      = {"attack", "clean_plain", "clean_lookalike"}
VALID_VARIANTS    = {"vi_full", "vi_nodiacritic", "mt_en", "bilingual", "mixed", "teencode"}
VALID_GOALS       = {"exfiltration", "redirect", "direct_harm"}
VALID_STYLES      = {"plain_command", "authority_spoof", "data_blend", "conditional", "metadata_spoof"}
VALID_POSITIONS   = {"start", "middle", "end"}


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def check_schema(records, filepath):
    errors = []
    for i, r in enumerate(records):
        # Thiếu trường
        for field in REQUIRED_FIELDS:
            if field not in r:
                errors.append(f"  [{i}] {r.get('id','?')} thiếu trường '{field}'")

        # Enum check
        if r.get("task_type") not in VALID_TASK_TYPES:
            errors.append(f"  [{i}] {r.get('id','?')} task_type lạ: {r.get('task_type')}")
        if r.get("group") not in VALID_GROUPS:
            errors.append(f"  [{i}] {r.get('id','?')} group lạ: {r.get('group')}")
        if r.get("variant") not in VALID_VARIANTS:
            errors.append(f"  [{i}] {r.get('id','?')} variant lạ: {r.get('variant')}")

        # Quy tắc null cho attack
        if r.get("is_benign") == False:
            if r.get("attacker_goal") not in VALID_GOALS:
                errors.append(f"  [{i}] {r.get('id','?')} attack nhưng attacker_goal lạ: {r.get('attacker_goal')}")
            if r.get("injection_style") not in VALID_STYLES:
                errors.append(f"  [{i}] {r.get('id','?')} attack nhưng injection_style lạ: {r.get('injection_style')}")
            if r.get("injection_position") not in VALID_POSITIONS:
                errors.append(f"  [{i}] {r.get('id','?')} attack nhưng injection_position lạ: {r.get('injection_position')}")
            if r.get("injected_instruction") is None:
                errors.append(f"  [{i}] {r.get('id','?')} attack nhưng injected_instruction=null")
            if r.get("success_condition") is None:
                errors.append(f"  [{i}] {r.get('id','?')} attack nhưng success_condition=null")
            # injection_span phải là [int, int] cho vi_full và vi_nodiacritic
            if r.get("variant") in ("vi_full", "vi_nodiacritic"):
                span = r.get("injection_span")
                if not (isinstance(span, list) and len(span) == 2 and all(isinstance(x, int) for x in span)):
                    errors.append(f"  [{i}] {r.get('id','?')} variant={r.get('variant')} nhưng injection_span không hợp lệ: {span}")

        # Quy tắc null cho clean
        if r.get("is_benign") == True:
            if r.get("injected_instruction") is not None:
                errors.append(f"  [{i}] {r.get('id','?')} benign nhưng injected_instruction không null")
            if r.get("success_condition") is not None:
                errors.append(f"  [{i}] {r.get('id','?')} benign nhưng success_condition không null")

    return errors


def main():
    all_records = []
    file_errors = {}
    total_files = 0

    print("=" * 60)
    print("  ViTransInject — Dataset Validation")
    print("=" * 60)

    # ── Load + schema check ───────────────────────────────────────
    print("\n[1/5] Schema check...\n")
    for rel_path, _ in DATA_FILES:
        full_path = os.path.join(BASE_DIR, rel_path)
        if not os.path.exists(full_path):
            print(f"  ⚠  MISSING: {rel_path}")
            continue

        records = load_jsonl(full_path)
        errors  = check_schema(records, rel_path)
        status  = "✓" if not errors else f"✗ ({len(errors)} lỗi)"
        print(f"  {status}  {rel_path}  ({len(records)} records)")
        if errors:
            for e in errors[:5]:
                print(e)
            if len(errors) > 5:
                print(f"    ... và {len(errors)-5} lỗi nữa")

        all_records.extend(records)
        file_errors[rel_path] = errors
        total_files += 1

    schema_ok = all(len(e) == 0 for e in file_errors.values())
    print(f"\n  Tổng: {len(all_records)} records từ {total_files} files")

    # ── Trùng id ──────────────────────────────────────────────────
    print("\n[2/5] Kiểm tra trùng id...")
    id_counter = collections.Counter(r["id"] for r in all_records if "id" in r)
    dupes = {k: v for k, v in id_counter.items() if v > 1}
    if dupes:
        print(f"  ✗  {len(dupes)} id bị trùng:")
        for k, v in list(dupes.items())[:10]:
            print(f"     {k} ({v} lần)")
    else:
        print(f"  ✓  Không có id trùng")

    # ── success_condition nhất quán ───────────────────────────────
    print("\n[3/5] Kiểm tra success_condition nhất quán qua variants...")
    attack_records = [r for r in all_records if r.get("group") == "attack"]
    by_seed = collections.defaultdict(list)
    for r in attack_records:
        by_seed[r["seed_id"]].append(r)

    inconsistent = []
    for seed_id, variants in by_seed.items():
        scs = []
        for r in variants:
            sc = r.get("success_condition")
            if sc is not None:
                scs.append(json.dumps(sc, sort_keys=True))
        unique_scs = set(scs)
        if len(unique_scs) > 1:
            inconsistent.append(seed_id)

    if inconsistent:
        print(f"  ✗  {len(inconsistent)} seed_id có success_condition không nhất quán:")
        for s in inconsistent[:5]:
            print(f"     {s}")
    else:
        print(f"  ✓  success_condition nhất quán ({len(by_seed)} seeds)")

    # ── Phân bổ ───────────────────────────────────────────────────
    print("\n[4/5] Phân bổ dataset...\n")
    attack_full = [r for r in all_records if r.get("group") == "attack" and r.get("variant") == "vi_full"]

    print("  task_type:")
    for k, v in sorted(collections.Counter(r["task_type"] for r in attack_full).items()):
        print(f"    {k}: {v}")

    print("  attacker_goal:")
    for k, v in sorted(collections.Counter(r.get("attacker_goal") for r in attack_full).items()):
        print(f"    {k}: {v}")

    print("  injection_style:")
    for k, v in sorted(collections.Counter(r.get("injection_style") for r in attack_full).items()):
        print(f"    {k}: {v}")

    print("  injection_position:")
    for k, v in sorted(collections.Counter(r.get("injection_position") for r in attack_full).items()):
        print(f"    {k}: {v}")

    print(f"\n  Tổng theo group:")
    for k, v in sorted(collections.Counter(r["group"] for r in all_records).items()):
        print(f"    {k}: {v}")

    print(f"\n  Tổng theo variant (attack only):")
    for k, v in sorted(collections.Counter(r["variant"] for r in attack_records).items()):
        print(f"    {k}: {v}")

    # ── needs_human_review ────────────────────────────────────────
    print("\n[5/5] needs_human_review...")
    review_records = [r for r in all_records if r.get("needs_human_review") == True]
    if review_records:
        by_variant = collections.Counter(r.get("variant") for r in review_records)
        print(f"  ⚠  {len(review_records)} records cần người duyệt:")
        for k, v in by_variant.items():
            print(f"    {k}: {v}")
    else:
        print("  ✓  Không có record cần duyệt")

    # ── Kết quả tổng ─────────────────────────────────────────────
    print("\n" + "=" * 60)
    passed = schema_ok and not dupes and not inconsistent
    if passed:
        print("  ✅  PASSED — Dataset sẵn sàng cho Bước 7 (split + đóng gói)")
    else:
        print("  ❌  FAILED — Cần sửa lỗi trước khi đóng gói")
    print("=" * 60)


if __name__ == "__main__":
    main()
