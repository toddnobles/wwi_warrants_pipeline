import json
import re
from typing import List, Dict

# ----------------------------
# Regexes tuned to RG 60
# ----------------------------

# Looks like an ID OR a date — we will disambiguate by name presence
ID_LINE_RE = re.compile(r"^\s*\d{1,4}-\d*")

# Name line: capitalized name followed by comma
NAME_LINE_RE = re.compile(r"^[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*,")

ID_EXTRACT_RE = re.compile(r"\b\d{1,4}-\d*\b")


# ----------------------------
# Helpers
# ----------------------------

def extract_ids(lines: List[str]) -> List[str]:
    ids = set()
    for l in lines:
        ids.update(ID_EXTRACT_RE.findall(l))
    return sorted(ids)


def extract_names(lines: List[str]) -> List[str]:
    names = set()
    for l in lines:
        if NAME_LINE_RE.match(l.strip()):
            names.add(l.split(",")[0].strip())
    return sorted(names)


def looks_like_person_start(lines: List[str], idx: int) -> bool:
    """
    A person starts at line idx if:
    - line looks like ID
    - a name line appears within the next 2 lines
    """
    if not ID_LINE_RE.match(lines[idx]):
        return False

    lookahead = lines[idx + 1: idx + 3]
    for la in lookahead:
        if NAME_LINE_RE.match(la.strip()):
            return True

    return False


# ----------------------------
# Main segmentation
# ----------------------------

def segment_people_from_jsonl(jsonl_path: str) -> List[Dict]:
    people = []
    person_index = 0

    with open(jsonl_path, "r") as f:
        for page_idx, raw in enumerate(f):
            record = json.loads(raw)
            text = record.get("text", "")
            lines = [l.rstrip() for l in text.splitlines() if l.strip()]

            current = []
            i = 0

            while i < len(lines):
                line = lines[i]

                if looks_like_person_start(lines, i) and current:
                    # finalize previous person
                    people.append({
                        "person_index": person_index,
                        "page_index": page_idx,
                        "raw_text": "\n".join(current),
                        "id_candidates": extract_ids(current),
                        "name_candidates": extract_names(current)
                    })
                    person_index += 1
                    current = []

                current.append(line)
                i += 1

            # finalize last person on page
            if current:
                people.append({
                    "person_index": person_index,
                    "page_index": page_idx,
                    "raw_text": "\n".join(current),
                    "id_candidates": extract_ids(current),
                    "name_candidates": extract_names(current)
                })
                person_index += 1

    return people

if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python segment_rg60.py <path_to_jsonl>")
        sys.exit(1)

    jsonl_path = sys.argv[1]

    segments = segment_people_from_jsonl(jsonl_path)

    print(f"\nExtracted {len(segments)} person blocks\n")

    # Print a preview of first 5 segments
    for seg in segments[:20]:
        print("=" * 60)
        print(f"Person index: {seg['person_index']}")
        print(f"ID candidates: {seg['id_candidates']}")
        print(f"Name candidates: {seg['name_candidates']}")
        print("TEXT PREVIEW:")
        print(seg["raw_text"][:500], "...\n")


print("\nBlocks missing names:")
for seg in segments:
    if not seg["name_candidates"]:
        print(f"⚠️ Person {seg['person_index']} | IDs: {seg['id_candidates']}")


with open("segmented_people.json", "w") as out:
    json.dump(segments, out, indent=2)