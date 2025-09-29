"""
This script processes drug leaflets in PDF format and extracts structured
biomedical knowledge triples (subject, relation, object) using an LLM model.
It applies majority voting across multiple generations to increase reliability,
assigns confidence scores, and saves results into a CSV file. The script also
keeps track of processed files to avoid duplication and enforces API rate limits.
"""

# === Imports ===
import requests
import re
import fitz
from collections import Counter
import csv
import os
import json
import time
import logging
import argparse

# === Smart Rate Control Settings (defaults; can be overridden by CLI) ===
MAX_REQUESTS_PER_MINUTE = 15
MAX_REQUESTS_PER_HOUR = 900
MAX_REQUESTS_PER_DAY = 2000
request_log = []

# === Rate Limit Enforcement ===
def enforce_rate_limit():
    """
    Enforces per-minute, per-hour, and per-day request limits to the API.
    If limits are exceeded, the function sleeps until limits reset.
    """
    global request_log
    now = time.time()
    
    # Clean up old timestamps
    request_log = [t for t in request_log if now - t < 86400]

    # Enforce per-minute limit
    recent_minute = [t for t in request_log if now - t < 60]
    if len(recent_minute) >= MAX_REQUESTS_PER_MINUTE:
        sleep_time = 60 - (now - recent_minute[0])
        log.info(f"[RATE LIMIT] Sleeping {sleep_time:.2f}s to respect per-minute limit...")
        time.sleep(sleep_time)

    # Enforce per-hour limit
    recent_hour = [t for t in request_log if now - t < 3600]
    if len(recent_hour) >= MAX_REQUESTS_PER_HOUR:
        sleep_time = 3600 - (now - recent_hour[0])
        log.info(f"[RATE LIMIT] Sleeping {sleep_time/60:.2f} mins to respect per-hour limit...")
        time.sleep(sleep_time)

    # Enforce per-day limit
    if len(request_log) >= MAX_REQUESTS_PER_DAY:
        sleep_time = 86400 - (now - request_log[0])
        log.info(f"[RATE LIMIT] Sleeping {sleep_time/3600:.2f} hrs to respect daily limit...")
        time.sleep(sleep_time)

    request_log.append(now)

# === Logging Configuration ===
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)
log = logging.getLogger()

# === Constants (set via CLI arguments) ===
PROCESSED_LOG = None
OUTPUT_CSV = None
PDF_DIR = None
CACHE_DIR = None
MAX_RETRIES = 3
RETRY_WAIT = 2
GEN_N = 5  # number of generations for majority voting

# API configuration (required via CLI)
API_URL = None
MODEL_NAME = None
API_KEY = None

# === Helper Functions ===

def extract_text_from_pdf(pdf_path):
    """Extracts raw text from a given PDF file using PyMuPDF (fitz)."""
    try:
        text = ""
        with fitz.open(pdf_path) as doc:
            for page in doc:
                text += page.get_text()
        return text
    except Exception as e:
        log.info(f"Failed to read PDF {pdf_path}: {e}")
        return ""

def query_model(prompt, temperature=0.5):
    """Queries the configured LLM API with a given prompt and returns model output."""
    api_url = API_URL
    model = MODEL_NAME

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant that extracts biomedical knowledge."},
            {"role": "user", "content": prompt}
        ],
        "temperature": temperature,
        "max_tokens": 1000
    }

    for attempt in range(MAX_RETRIES):
        try:
            enforce_rate_limit() 
            response = requests.post(api_url, headers=headers, json=payload, timeout=300)
            if response.status_code != 200:
                log.warning(f"Attempt {attempt+1}/{MAX_RETRIES} - API returned status {response.status_code}, retrying...")
                time.sleep(RETRY_WAIT)
                continue

            data = response.json()
            if "choices" not in data:
                log.warning(f"Attempt {attempt+1}/{MAX_RETRIES} - Unexpected response format: {data}")
                return ""

            return data["choices"][0]["message"]["content"]

        except Exception as e:
            log.warning(f"Attempt {attempt+1}/{MAX_RETRIES} - Exception during API call: {e}")
            time.sleep(RETRY_WAIT)

    log.error("Max retries exceeded for Chat-AI model.")
    return ""

def extract_triples(text):
    """Extracts triples from model text output using regex matching."""
    lines = text.strip().split('\n')
    triples = []
    for line in lines:
        match = re.match(r"\(?\s*([^,]+?)\s*,\s*([^,]+?)\s*,\s*([^)]+?)\s*\)?\.?$", line.strip())
        if match:
            triples.append(tuple(part.strip() for part in match.groups()))
    return triples

def normalize(triple):
    """Normalizes a triple by lowercasing and stripping whitespace."""
    return tuple(part.lower().strip() for part in triple)

def extract_all_triples_with_counts(text, pdf_id, n=5):
    """
    Generates triples n times from the LLM, applies normalization and deduplication,
    and returns triples with counts and confidence scores.
    """
    prompt = f"""
You are a biomedical extraction system.

Extract triples from the drug leaflet below in the exact format:
(subject, predicate, object)

### Format Rules:
- No explanations, No reasoning. Just generate the triples.
- Each triple must be on a separate line.
- Only use the following predicates:
  hasSideEffect, hasWarning, hasContraindication, hasActiveIngredient,
  hasInactiveIngredient, hasDosageInfo, hasStorageInfo, hasShape, hasColour
- Use the drug name from the leaflet as the subject.
- Object must be short, atomic.

### Example:
(noradrenaline, hasSideEffect, headache)
(noradrenaline, hasStorageInfo, protect from light)

### Leaflet:
\"\"\"{text}\"\"\"
""".strip()

    cache_file = os.path.join(CACHE_DIR, f"{pdf_id}.json")
    all_triples = []

    # Resume from cache if exists
    if os.path.exists(cache_file):
        with open(cache_file, "r") as f:
            all_triples = json.load(f)
        log.info(f"Resuming {pdf_id}: {len(all_triples)} generations already completed.")

    current_gen = len(all_triples)
    temperatures = [0.5] * n

    while current_gen < n:
        temp = temperatures[current_gen]
        log.info(f"\n GENERATION {current_gen+1} (temperature={temp})")
        try:
            output = query_model(prompt, temperature=temp)
            if not output:
                log.warning(f"No output from model for {pdf_id}")
                continue
            triples = extract_triples(output)
            normalized = [normalize(tr) for tr in triples if len(tr) == 3]
            all_triples.append(list(set(normalized)))
            current_gen += 1
            log.info(f"GEN {current_gen} - {len(normalized)} triples")
        except Exception as e:
            log.info(f"Retry {current_gen+1}/{MAX_RETRIES} failed: {e}")
            time.sleep(RETRY_WAIT)

        # Save progress
        with open(cache_file, "w") as f:
            json.dump(all_triples, f)

    flattened = [tr for sublist in all_triples for tr in sublist]
    triple_counts = Counter(flattened)

    triple_with_counts = [
        (tr[0], tr[1], tr[2], pdf_id, count, round(min(count/float(n), 1.0), 2))
        for tr, count in triple_counts.items()
    ]

    fully_completed = current_gen == n
    return triple_with_counts, fully_completed

def get_processed_files():
    """Loads the set of already processed files from log."""
    if not os.path.exists(PROCESSED_LOG):
        return set()
    with open(PROCESSED_LOG, "r") as f:
        return set(line.strip() for line in f.readlines())

def mark_as_processed(filename):
    """Marks a file as processed by appending to the processed log."""
    with open(PROCESSED_LOG, "a") as f:
        f.write(filename + "\n")

def save_triples_to_csv(triples):
    """Appends extracted triples to the output CSV file."""
    write_header = not os.path.exists(OUTPUT_CSV)
    with open(OUTPUT_CSV, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["Subject", "Relation", "Object", "Filename", "Count", "Confidence"])
        writer.writerows(triples)

# === Main Function ===
def main():
    """Main pipeline: loads PDFs, extracts triples, saves results."""
    processed = get_processed_files()
    pdf_files = [f for f in os.listdir(PDF_DIR) if f.endswith(".pdf")]
    total_files = len(pdf_files)

    for idx, pdf_file in enumerate(pdf_files, start=1):
        if pdf_file in processed:
            log.info(f"Skipping already processed: {pdf_file}")
            continue

        log.info(f"\nProcessing {idx}/{total_files}: {pdf_file}")
        full_path = os.path.join(PDF_DIR, pdf_file)
        text = extract_text_from_pdf(full_path)

        if not text.strip():
            log.info(f"Empty text extracted from {pdf_file}. Skipping.")
            continue

        try:
            pdf_id = os.path.splitext(pdf_file)[0]
            triples_with_counts, fully_completed = extract_all_triples_with_counts(text, pdf_id, n=GEN_N)

            if not fully_completed:
                log.info(f"Not all generations completed for {pdf_file}. Will retry later.")
                continue

            log.info(f"Extracted {len(triples_with_counts)} triples (with counts).")
            save_triples_to_csv(triples_with_counts)
            mark_as_processed(pdf_file)

            # Clean up cache
            cache_path = os.path.join(CACHE_DIR, f"{pdf_id}.json")
            if os.path.exists(cache_path):
                os.remove(cache_path)

        except Exception as e:
            log.info(f"Error processing {pdf_file}: {e}")

# === CLI Argument Parser ===
def parse_args():
    """Parses CLI arguments for reproducible pipeline execution."""
    p = argparse.ArgumentParser(description="KG triple extraction with majority voting.")
    # --- Mandatory: File paths + API details ---
    p.add_argument("--pdf-dir", required=True, help="Directory containing PDFs to process.")
    p.add_argument("--output-csv", required=True, help="Output CSV path.")
    p.add_argument("--processed-log", required=True, help="Path to processed-files log.")
    p.add_argument("--cache-dir", required=True, help="Directory for generation cache.")
    p.add_argument("--api-url", required=True, help="Chat API URL.")
    p.add_argument("--model-name", required=True, help="Model name to use.")
    p.add_argument("--api-key", required=True, help="API key (or set via env and pass here).")

    # --- Optional with defaults ---
    p.add_argument("--max-retries", type=int, default=3, help="Max retries for API calls.")
    p.add_argument("--retry-wait", type=int, default=2, help="Seconds to wait between retries.")
    p.add_argument("--rpm", type=int, default=15, help="Max requests per minute.")
    p.add_argument("--rph", type=int, default=900, help="Max requests per hour.")
    p.add_argument("--rpd", type=int, default=21600, help="Max requests per day.")
    p.add_argument("--gens", type=int, default=5, help="Number of generations per PDF (majority voting).")
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG","INFO","WARNING","ERROR","CRITICAL"],
                   help="Logging level.")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()

    # Override globals with CLI values
    PROCESSED_LOG = args.processed_log
    OUTPUT_CSV = args.output_csv
    PDF_DIR = args.pdf_dir
    CACHE_DIR = args.cache_dir
    MAX_RETRIES = args.max_retries
    RETRY_WAIT = args.retry_wait
    MAX_REQUESTS_PER_MINUTE = args.rpm
    MAX_REQUESTS_PER_HOUR = args.rph
    MAX_REQUESTS_PER_DAY = args.rpd
    GEN_N = args.gens

    API_URL = args.api_url
    MODEL_NAME = args.model_name
    API_KEY = args.api_key

    # Apply logging level and ensure cache directory exists
    log.setLevel(getattr(logging, args.log_level))
    os.makedirs(CACHE_DIR, exist_ok=True)

    main()
