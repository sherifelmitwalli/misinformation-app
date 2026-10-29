import os
import sys
import json
import requests
from dotenv import load_dotenv

# Set UTF-8 encoding
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

# Inject Windows enterprise certificates
import truststore
truststore.inject_into_ssl()

import httpx
_orig_sync_init = httpx.Client.__init__
def _patched_sync_init(self, *args, **kwargs):
    headers = kwargs.get('headers') or {}
    if isinstance(headers, dict):
        headers['Accept-Encoding'] = 'gzip, deflate'
    kwargs['headers'] = headers
    _orig_sync_init(self, *args, **kwargs)
httpx.Client.__init__ = _patched_sync_init

load_dotenv()
from openai import OpenAI

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SERPER_API_KEY = os.getenv("SERPER_API_KEY")

custom_http = httpx.Client(headers={"Accept-Encoding": "gzip, deflate"})
client = OpenAI(api_key=OPENAI_API_KEY, http_client=custom_http)

MODEL_NAME = "gpt-4o"

PHRASINGS = [
    {
        "id": "A",
        "phrasing": "Meetings of WHO FCTC conference of the parties are biased against harm reduction.",
        "framing": "Broad Evaluative Allegation (John's original claim)"
    },
    {
        "id": "B",
        "phrasing": "The WHO FCTC treaty excludes harm reduction from tobacco control strategies.",
        "framing": "Absolute Legal Assertion (Factually verifiable against treaty text)"
    },
    {
        "id": "C",
        "phrasing": "The WHO FCTC COP ignores evidence on e-cigarettes and blocks smokers from quitting.",
        "framing": "Industry / Advocacy Rhetoric (Aggressive emotional framing)"
    },
    {
        "id": "D",
        "phrasing": "The WHO FCTC adopts a precautionary approach toward novel nicotine products due to youth addiction risks and evolving toxicology.",
        "framing": "Institutional / Precautionary Framing (Official WHO / COP position)"
    },
    {
        "id": "E",
        "phrasing": "Tobacco companies use harm reduction rhetoric to lobby against FCTC regulations and promote novel nicotine products to youth.",
        "framing": "Public Health / TCRG Research Framing (Industry tactics focus)"
    }
]

def search_serper(query):
    url = "https://google.serper.dev/search"
    headers = {
        "X-API-KEY": SERPER_API_KEY,
        "Content-Type": "application/json"
    }
    payload = json.dumps({"q": query, "num": 4})
    try:
        response = requests.post(url, headers=headers, data=payload, timeout=15)
        if response.status_code == 200:
            data = response.json()
            return [item.get("snippet", "") for item in data.get("organic", [])[:3]]
    except Exception as e:
        pass
    return []

def classify_score(score):
    if score >= 81.0:
        return "Highly likely to be credible"
    elif score >= 61.0:
        return "Likely to be credible"
    elif score >= 41.0:
        return "Moderate credibility"
    elif score >= 21.0:
        return "Unlikely to be credible"
    else:
        return "Highly unlikely to be credible"

def evaluate_phrasing(item):
    print(f"\nEvaluating Variant {item['id']} ({item['framing']}):")
    print(f"  Claim: \"{item['phrasing']}\"")
    
    snippets = search_serper(f"WHO FCTC harm reduction {item['phrasing'][:50]}")
    context = "\n".join(snippets)
    
    prompt = f"""
You are the Health Evidence Assessor in the multi-agent AI pipeline for tobacco misinformation (Elmitwalli et al., Frontiers in AI, 2025).

Evaluate the following claim variation:
CLAIM: "{item['phrasing']}"
CLAIM FRAMING: {item['framing']}

RETRIEVED CONTEXT:
{context}

KEY KNOWLEDGE:
- FCTC Article 1(d) explicitly includes harm reduction in the definition of tobacco control.
- COP decisions apply the precautionary principle to novel products (ENDS/HTPs) due to youth uptake and incomplete independent evidence.
- Transnational tobacco companies (TTCs) documentedly exploit harm reduction narratives to lobby against strict advertising and flavor bans (TCRG / STOP research).

Score the claim strictly on how well available authoritative evidence and treaty text SUPPORT this specific phrasing (0 to 100):
- Eq (Evidence Quality, 40%): Rigor of evidence supporting the claim.
- Sc (Scientific Consensus, 25%): Degree of consensus in favor of the claim.
- Ec (Evidence Consistency, 15%): Consistency across studies supporting the claim.
- Er (Evidence Recency, 10%): Recency and currency of supporting evidence.
- Sp (Scientific / Legal Plausibility, 10%): Plausibility under international treaty law.

Output strictly JSON:
{{
  "Eq": integer,
  "Sc": integer,
  "Ec": integer,
  "Er": integer,
  "Sp": integer,
  "short_justification": "2-sentence explanation of why this specific phrasing yields this score."
}}
"""

    resp = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": "You are a public health policy evaluator. Output valid JSON only."},
            {"role": "user", "content": prompt}
        ],
        response_format={"type": "json_object"},
        temperature=0.0
    )
    
    data = json.loads(resp.choices[0].message.content)
    eq = float(data["Eq"])
    sc = float(data["Sc"])
    ec = float(data["Ec"])
    er = float(data["Er"])
    sp = float(data["Sp"])
    
    # Deterministic Algorithm 3 calculation
    final_score = round((eq * 0.40) + (sc * 0.25) + (ec * 0.15) + (er * 0.10) + (sp * 0.10), 2)
    level = classify_score(final_score)
    
    print(f"  Result: Score={final_score}/100 [{level}] (Eq={eq}, Sc={sc}, Ec={ec}, Er={er}, Sp={sp})")
    
    return {
        "id": item["id"],
        "phrasing": item["phrasing"],
        "framing": item["framing"],
        "scores": {"Eq": eq, "Sc": sc, "Ec": ec, "Er": er, "Sp": sp},
        "final_score": final_score,
        "classification": level,
        "justification": data["short_justification"]
    }

def main():
    print("=" * 80)
    print("TESTING HARM REDUCTION PHRASING SENSITIVITY IN MULTI-AGENT PIPELINE")
    print("=" * 80)
    
    results = []
    for p in PHRASINGS:
        res = evaluate_phrasing(p)
        results.append(res)
    
    with open("harm_reduction_phrasing_experiment.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "=" * 80)
    print("EXPERIMENT COMPLETE: SAVED TO harm_reduction_phrasing_experiment.json")

if __name__ == "__main__":
    main()
