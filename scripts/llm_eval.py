"""
Reads an input TXT file containing triples grouped by filename, loads the
corresponding leaflet PDFs, and uses an LLM to judge each triple as
Correct / Incorrect / Partially Correct with reasoning. Results are written
to a CSV.

All paths and API configuration must be provided by the user.
"""

# =========================
# Imports
# =========================
import os
import re
import csv
import json
import fitz  # PyMuPDF
import requests
import argparse
import sys
from typing import List, Dict, Any, Tuple, Set

# =========================
# Globals (set via CLI)
# =========================
PDF_SUFFIX = None
TEMPERATURE = None
MAX_TOKENS = None
MAX_TRIPLES_PER_CALL = None
MAX_LEAFLET_CHARS = None

ALLOWED_LABELS = {"CORRECT", "INCORRECT", "PARTIALLY CORRECT"}

API_URL = None
MODEL_NAME = None
API_KEY = None

_session = requests.Session()

# =========================
# LLM query
# =========================
def query_model(prompt: str,
                      temperature: float,
                      max_tokens: int) -> str:
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system",
             "content": "You validate biomedical triples strictly against the leaflet. Be terse and precise."},
            {"role": "user", "content": prompt},
        ],
        "temperature": float(temperature),
        "max_tokens": int(max_tokens),
    }
    resp = _session.post(API_URL, headers=headers, json=payload)
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"]

# =========================
# Prompting rules
# =========================
SYSTEM_RULES = """You will validate biomedical triples against a drug leaflet's full text.
For EACH triple, output exactly one label:
- Correct
- Incorrect
- Partially Correct

Reasoning:
- Provide a VERY short justification grounded in the leaflet (≤25 words).
- IMPORTANT: Do NOT use double-quote characters ("). If quoting, use single quotes (') or paraphrase.
- No newlines inside reasoning (single line).

Return JSON ONLY in this schema:
[
  {"index": <1-based index>, "label": "Correct|Incorrect|Partially Correct", "reasoning": "<no double quotes, one line>"}
]
"""

# =========================
# Helpers
# =========================
def parse_input_txt(path: str) -> List[Tuple[str, List[Dict[str, str]]]]:
    blocks = []
    current_filename = None
    current_triples = []

    filename_re = re.compile(r"^\s*Filename:\s*(.+?)\s*$", re.IGNORECASE)
    triple_re = re.compile(r"^\s*-\s*(.+?)\s*\|\s*(.+?)\s*\|\s*(.+?)\s*$")

    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line.strip():
                continue

            m_fn = filename_re.match(line)
            if m_fn:
                if current_filename is not None:
                    blocks.append((current_filename, current_triples))
                current_filename = m_fn.group(1).strip()
                current_triples = []
                continue

            m_tr = triple_re.match(line)
            if m_tr and current_filename is not None:
                subj = m_tr.group(1).strip()
                rel = m_tr.group(2).strip().lower()
                obj = m_tr.group(3).strip()
                current_triples.append({"subject": subj, "relation": rel, "object": obj})

    if current_filename is not None:
        blocks.append((current_filename, current_triples))

    return blocks

def extract_pdf_text_full(pdf_path: str) -> str:
    doc = fitz.open(pdf_path)
    parts = [doc.load_page(i).get_text("text") for i in range(len(doc))]
    doc.close()
    return "\n".join(parts).replace("\u00A0", " ")

def sanitize_for_filename(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s)[:120]

def build_prompt(leaflet_text: str, triples: List[Dict[str, Any]]) -> str:
    triples_lines = []
    for i, t in enumerate(triples, start=1):
        triples_lines.append(
            f'{i}. Subject="{t["subject"]}" | Relation="{t["relation"]}" | Object="{t["object"]}"'
        )

    if MAX_LEAFLET_CHARS and len(leaflet_text) > MAX_LEAFLET_CHARS:
        leaflet_text = leaflet_text[:MAX_LEAFLET_CHARS]

    return (
        "You are given the FULL drug leaflet text, followed by triples to validate.\n\n"
        "=== BEGIN LEAFLET TEXT ===\n"
        f"{leaflet_text}\n"
        "=== END LEAFLET TEXT ===\n\n"
        "=== TRIPLES TO EVALUATE (IN ORDER) ===\n"
        + "\n".join(triples_lines)
        + "\n\n"
        + SYSTEM_RULES
    )

def normalize_label(lbl: str) -> str:
    s = str(lbl).strip().upper()
    if s in ALLOWED_LABELS:
        return "Correct" if s == "CORRECT" else \
               "Incorrect" if s == "INCORRECT" else "Partially Correct"
    return "Incorrect"

def _strip_code_fences(s: str) -> str:
    m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", s, flags=re.IGNORECASE)
    return m.group(1) if m else s

def _extract_json_array(s: str) -> str:
    s = s.strip()
    start, end = s.find("["), s.rfind("]")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON array found in model output.")
    return s[start:end + 1]

def parse_model_list(raw: str):
    cleaned = _strip_code_fences(raw)
    cleaned = cleaned.replace("“", '"').replace("”", '"').replace("’", "'")
    cleaned = _extract_json_array(cleaned)
    return json.loads(cleaned)

# =========================
# Evaluation per file
# =========================
def judge_one_file(filename: str,
                   triples: List[Dict[str, Any]],
                   pdf_dir: str) -> List[List[str]]:
    rows = []
    pdf_path = os.path.join(pdf_dir, filename + PDF_SUFFIX)

    if not os.path.exists(pdf_path):
        for t in triples:
            rows.append([filename, t["subject"], t["relation"], t["object"], "", "", f"PDF not found: {pdf_path}"])
        return rows

    try:
        leaflet_text = extract_pdf_text_full(pdf_path)
    except Exception as e:
        err = f"PDF read error: {e}"
        for t in triples:
            rows.append([filename, t["subject"], t["relation"], t["object"], "", "", err])
        return rows

    for start in range(0, len(triples), MAX_TRIPLES_PER_CALL):
        chunk = triples[start:start + MAX_TRIPLES_PER_CALL]
        prompt = build_prompt(leaflet_text, chunk)

        try:
            content = query_model(prompt, temperature=TEMPERATURE, max_tokens=MAX_TOKENS)
            parsed = parse_model_list(content)
        except Exception as e:
            err = f"LLM error: {e}"
            for t in chunk:
                rows.append([filename, t["subject"], t["relation"], t["object"], "", "", err])
            continue

        by_idx = {}
        for i, obj in enumerate(parsed):
            idx_key = int(obj.get("index", i + 1)) if obj.get("index") else i + 1
            by_idx[idx_key] = obj

        for i, t in enumerate(chunk, start=1):
            obj = by_idx.get(i, {})
            label = normalize_label(obj.get("label", "Incorrect"))
            reasoning = str(obj.get("reasoning", "")).replace("\n", " ").replace('"', "'").strip()
            rows.append([filename, t["subject"], t["relation"], t["object"], label, reasoning, ""])

    return rows

# =========================
# Helpers
# =========================
def get_processed_filenames(out_csv: str) -> Set[str]:
    processed = set()
    if not os.path.exists(out_csv) or os.path.getsize(out_csv) == 0:
        return processed
    with open(out_csv, "r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            fn = (row.get("Filename") or "").strip()
            if fn:
                processed.add(fn)
    return processed

def open_output_for_resume(out_csv: str):
    is_new = not os.path.exists(out_csv) or os.path.getsize(out_csv) == 0
    mode = "a" if not is_new else "w"
    fh = open(out_csv, mode, newline="", encoding="utf-8")
    writer = csv.writer(fh)
    if is_new:
        writer.writerow(["Filename", "Subject", "Relation", "Object", "Label", "Reasoning", "Error"])
        fh.flush()
    return fh, writer

# =========================
# Main Evaluator Function
# =========================
def run_llm_judge_from_txt(input_txt: str, pdf_dir: str, out_csv: str):
    blocks = parse_input_txt(input_txt)
    already = get_processed_filenames(out_csv)
    fh, writer = open_output_for_resume(out_csv)

    try:
        for filename, triples in blocks:
            if filename in already:
                print(f"[resume] Skipping already processed: {filename}")
                continue
            rows = judge_one_file(filename=filename, triples=triples, pdf_dir=pdf_dir)
            writer.writerows(rows)
            fh.flush()
            already.add(filename)
    finally:
        fh.close()
    print(f"Done. Judgments saved to: {out_csv}")

# =========================
# CLI
# =========================
def parse_args():
    p = argparse.ArgumentParser(description="LLM-as-Judge: validate triples against leaflet PDFs.")
    p.add_argument("--input-txt", required=True, help="TXT file with triples grouped by filename.")
    p.add_argument("--pdf-dir", required=True, help="Directory containing PDF leaflets.")
    p.add_argument("--out-csv", required=True, help="Output CSV path (resume supported).")

    # No defaults: all must be given
    p.add_argument("--pdf-suffix", required=True, help="Suffix for PDF files (e.g., .pdf).")
    p.add_argument("--temperature", type=float, required=True, help="LLM temperature.")
    p.add_argument("--max-tokens", type=int, required=True, help="LLM max tokens in response.")
    p.add_argument("--max-triples-per-call", type=int, required=True, help="Triples per LLM call batch.")
    p.add_argument("--max-leaflet-chars", type=int, required=True, help="Max chars of leaflet text (0=disable).")

    p.add_argument("--api-url", required=True, help="Chat API URL.")
    p.add_argument("--model-name", required=True, help="Model name.")
    p.add_argument("--api-key", required=True, help="API key.")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()

    PDF_SUFFIX = args.pdf_suffix
    TEMPERATURE = args.temperature
    MAX_TOKENS = args.max_tokens
    MAX_TRIPLES_PER_CALL = args.max_triples_per_call
    MAX_LEAFLET_CHARS = args.max_leaflet_chars

    API_URL = args.api_url
    MODEL_NAME = args.model_name
    API_KEY = args.api_key

    run_llm_judge_from_txt(
        input_txt=args.input_txt,
        pdf_dir=args.pdf_dir,
        out_csv=args.out_csv,
    )
