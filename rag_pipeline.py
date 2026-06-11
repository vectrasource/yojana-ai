"""
Yojana AI — RAG Pipeline
========================
Run ONCE to build the search index from schemes data.
Then use search_schemes() in your app.

Usage:
  python3 rag_pipeline.py --build --file schemes.json
  python3 rag_pipeline.py --search "32 year old farmer in Kerala, income 80000"
  python3 rag_pipeline.py --stats
"""

import json
import csv
import os
import sys
import argparse
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ── Config ───────────────────────────────────────────────────────────────────
DB_PATH = "./yojana_db"
DB_FILE = os.path.join(DB_PATH, "schemes_index.pkl")
# TF-IDF search — works offline, no model download needed
# On AMD MI300X — upgrade TfidfVectorizer to sentence-transformers for semantic search

# ── Load Data ─────────────────────────────────────────────────────────────────
def load_json(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    elif isinstance(data, dict) and 'data' in data:
        return data['data']
    return [data]

def load_csv(filepath):
    schemes = []
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            schemes.append(dict(row))
    return schemes

def normalize_scheme(scheme):
    return {
        'name': str(scheme.get('scheme_name') or scheme.get('name') or scheme.get('title') or 'Unknown').strip(),
        'description': str(scheme.get('description') or scheme.get('details') or '').strip(),
        'eligibility': str(scheme.get('eligibility') or scheme.get('eligibility_criteria') or '').strip(),
        'benefits': str(scheme.get('benefits') or scheme.get('benefit') or '').strip(),
        'ministry': str(scheme.get('ministry') or scheme.get('department') or '').strip(),
        'state': str(scheme.get('state') or scheme.get('state_name') or 'Central').strip(),
        'url': str(scheme.get('url') or scheme.get('official_link') or scheme.get('applyUrl') or 'https://www.myscheme.gov.in').strip(),
        'application': str(scheme.get('application_process') or scheme.get('how_to_apply') or '').strip(),
        'tags': str(scheme.get('tags') or '').strip(),
    }

def build_document_text(s):
    parts = [
        f"Scheme {s['name']}",
        f"Ministry {s['ministry']}",
        f"State {s['state']}",
        f"Description {s['description']}",
        f"Eligibility {s['eligibility']}",
        f"Benefits {s['benefits']}",
        str(s.get('tags', '')),
    ]
    return ' '.join([p for p in parts if p.strip()])

# ── Build Index ───────────────────────────────────────────────────────────────
def build_database(filepath):
    print(f"\n📂 Loading schemes from: {filepath}")
    ext = os.path.splitext(filepath)[1].lower()
    if ext == '.json':
        raw = load_json(filepath)
    elif ext in ['.csv', '.tsv']:
        raw = load_csv(filepath)
    else:
        print("❌ Use .json or .csv"); sys.exit(1)

    print(f"✅ Loaded {len(raw)} schemes")
    schemes = [normalize_scheme(s) for s in raw]
    schemes = [s for s in schemes if s['name'] != 'Unknown' and len(s['description']) > 10]
    print(f"✅ {len(schemes)} valid schemes after filtering")

    print(f"\n🔨 Building TF-IDF search index...")
    documents = [build_document_text(s) for s in schemes]
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=10000, stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(documents)

    os.makedirs(DB_PATH, exist_ok=True)
    with open(DB_FILE, 'wb') as f:
        pickle.dump({'schemes': schemes, 'vectorizer': vectorizer, 'tfidf_matrix': tfidf_matrix}, f)

    print(f"✅ Database built! {len(schemes)} schemes indexed.")
    print(f"📁 Saved to: {DB_FILE}")

# ── Search ────────────────────────────────────────────────────────────────────
def search_schemes(user_profile: str, n_results: int = 6, state_filter: str = None):
    if not os.path.exists(DB_FILE):
        print("❌ Database not found. Run --build first.")
        return []

    with open(DB_FILE, 'rb') as f:
        db = pickle.load(f)

    schemes = db['schemes']
    vectorizer = db['vectorizer']
    tfidf_matrix = db['tfidf_matrix']

    query_vec = vectorizer.transform([user_profile])
    scores = cosine_similarity(query_vec, tfidf_matrix)[0]

    # Apply state filter if given
    results = []
    for i, score in enumerate(scores):
        s = schemes[i]
        state = s.get('state', 'Central')
        if state_filter:
            if state not in ['Central', 'All States', state_filter]:
                score *= 0.3  # Penalize non-matching states
        results.append((score, s))

    results.sort(key=lambda x: x[0], reverse=True)
    top = results[:n_results]

    output = []
    for score, s in top:
        pct = round(float(score) * 100, 1)
        scheme = dict(s)
        scheme['match_score'] = pct
        scheme['match_label'] = 'High' if pct > 15 else 'Medium' if pct > 5 else 'Low'
        output.append(scheme)
    return output

# ── Format for AI prompt ──────────────────────────────────────────────────────
def format_for_prompt(schemes: list) -> str:
    if not schemes:
        return "No matching schemes found in database."
    lines = ["RELEVANT GOVERNMENT SCHEMES FROM DATABASE:\n"]
    for i, s in enumerate(schemes, 1):
        lines.append(f"{i}. {s['name']} ({s.get('match_label','?')} Match)")
        lines.append(f"   Ministry: {s.get('ministry','N/A')} | State: {s.get('state','Central')}")
        lines.append(f"   Eligibility: {s.get('eligibility','N/A')[:200]}")
        lines.append(f"   Benefits: {s.get('benefits','N/A')[:200]}")
        lines.append(f"   Apply: {s.get('url','https://myscheme.gov.in')}")
        lines.append("")
    return '\n'.join(lines)

# ── Stats ─────────────────────────────────────────────────────────────────────
def show_stats():
    if not os.path.exists(DB_FILE):
        print("❌ No database found. Run --build first.")
        return
    with open(DB_FILE, 'rb') as f:
        db = pickle.load(f)
    print(f"\n📊 Database Stats:")
    print(f"   Total schemes indexed: {len(db['schemes'])}")
    print(f"   Location: {DB_FILE}")

# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Yojana AI RAG Pipeline')
    parser.add_argument('--build', action='store_true')
    parser.add_argument('--file', type=str)
    parser.add_argument('--search', type=str)
    parser.add_argument('--state', type=str)
    parser.add_argument('--stats', action='store_true')
    parser.add_argument('--n', type=int, default=6)
    args = parser.parse_args()

    if args.build:
        if not args.file or not os.path.exists(args.file):
            print("❌ Provide valid --file path"); sys.exit(1)
        build_database(args.file)

    elif args.search:
        print(f"\n🔍 Searching: {args.search}")
        results = search_schemes(args.search, n_results=args.n, state_filter=args.state)
        if results:
            print(f"\n✅ Top {len(results)} matching schemes:\n")
            for i, s in enumerate(results, 1):
                print(f"{i}. {s['name']} — {s['match_label']} ({s['match_score']}%)")
                print(f"   {s.get('benefits','')[:100]}...")
                print(f"   {s.get('url','')}\n")
        else:
            print("No results found.")

    elif args.stats:
        show_stats()
    else:
        parser.print_help()
