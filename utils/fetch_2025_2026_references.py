import os
import sys
import json
import urllib.request
import urllib.parse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REF_DIR = PROJECT_ROOT / "reference_papers"
REF_DIR.mkdir(parents=True, exist_ok=True)

queries = [
    'TITLE:"skin lesion" AND PUB_YEAR:2025',
    'TITLE:"skin lesion" AND PUB_YEAR:2026',
    'TITLE:"skin disease" AND PUB_YEAR:2025',
    'TITLE:"skin disease" AND PUB_YEAR:2026',
    'ABSTRACT:"skin lesion classification" AND PUB_YEAR:2025',
    'ABSTRACT:"skin lesion classification" AND PUB_YEAR:2026',
    'ABSTRACT:"skin cancer" AND ABSTRACT:"deep learning" AND PUB_YEAR:2025',
    'ABSTRACT:"skin cancer" AND ABSTRACT:"deep learning" AND PUB_YEAR:2026'
]

def fetch_papers():
    papers = []
    seen_titles = set()

    for q in queries:
        encoded_q = urllib.parse.quote(q)
        url = f'https://www.ebi.ac.uk/europepmc/webservices/rest/search?query={encoded_q}&format=json&pageSize=30'
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req) as res:
                data = json.loads(res.read().decode('utf-8'))
                results = data.get('resultList', {}).get('result', [])
                for r in results:
                    t = r.get('title', '').strip().rstrip('.')
                    if not t or t.lower() in seen_titles:
                        continue
                    # Skip unrelated medical domains if title is clearly not about skin classification
                    if 'spinal' in t.lower() or 'breast' in t.lower() or 'tumors and a pigmented' in t.lower():
                        continue
                    seen_titles.add(t.lower())
                    y = r.get('pubYear', '')
                    a = r.get('authorString', 'Unknown Authors')
                    j = r.get('journalTitle', r.get('bookTitle', 'IEEE / Springer / Elsevier'))
                    doi = r.get('doi', '')
                    pmid = r.get('pmid', '')
                    pmcid = r.get('pmcid', '')
                    abstract = r.get('abstractText', 'No abstract available.')
                    
                    papers.append({
                        'title': t,
                        'authors': a,
                        'year': y,
                        'journal': j,
                        'doi': doi,
                        'pmid': pmid,
                        'pmcid': pmcid,
                        'abstract': abstract
                    })
        except Exception as e:
            print(f"Error querying {q}: {e}")

    print(f"Total 2025/2026 skin disease classification papers collected: {len(papers)}")
    return papers

if __name__ == "__main__":
    papers = fetch_papers()
    json_path = REF_DIR / "references_2025_2026.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(papers, f, indent=2, ensure_ascii=False)
    print(f"Saved JSON database to {json_path}")
