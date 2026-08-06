import json


def load_jsonl(path: str) -> list:
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def filter_records(records: list, variant=None, task=None, group=None, limit=None) -> list:
    if variant:
        records = [r for r in records if r["variant"] == variant]
    if task:
        records = [r for r in records if r["task_type"] == task]
    if group:
        records = [r for r in records if r["group"] == group]
    if limit:
        records = records[:limit]
    return records
