import json
from pathlib import Path
import re

input_dir = Path("data/json")
output_file = Path("data/combined.jsonl")

records = []

# ---- Read all records ----
for jsonl_path in sorted(input_dir.glob("*.jsonl")):
    with jsonl_path.open("r", encoding="utf-8") as in_f:
        for line in in_f:
            if not line.strip():
                continue
            record = json.loads(line)

            # keep original jsonl filename if desired
            record["source_file"] = jsonl_path.name

            records.append(record)

# ---- Sort by metadata["Source-File"] ----
PAGE_RE = re.compile(r"page_(\d+)", re.IGNORECASE)

def source_file_sort_key(r):
    sf = r.get("metadata", {}).get("Source-File", "")

    m = PAGE_RE.search(sf)
    page = int(m.group(1)) if m else -1

    # remove page suffix for volume-level grouping
    volume = PAGE_RE.sub("", sf).strip()

    return (volume, page)

records.sort(key=source_file_sort_key)

# ---- Write combined, sorted JSONL ----
with output_file.open("w", encoding="utf-8") as out_f:
    for record in records:
        out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
