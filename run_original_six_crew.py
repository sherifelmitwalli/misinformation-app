import os
import sys
import re
import json

# Configure UTF-8 encoding for Windows console
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

# Inject Windows enterprise/institutional certificates
import truststore
truststore.inject_into_ssl()

# Patch HTTPX to avoid zstd decoding bug on Python 3.13
import httpx
_orig_sync_init = httpx.Client.__init__
def _patched_sync_init(self, *args, **kwargs):
    headers = kwargs.get('headers') or {}
    if isinstance(headers, dict):
        headers['Accept-Encoding'] = 'gzip, deflate'
    kwargs['headers'] = headers
    _orig_sync_init(self, *args, **kwargs)
httpx.Client.__init__ = _patched_sync_init

_orig_async_init = httpx.AsyncClient.__init__
def _patched_async_init(self, *args, **kwargs):
    headers = kwargs.get('headers') or {}
    if isinstance(headers, dict):
        headers['Accept-Encoding'] = 'gzip, deflate'
    kwargs['headers'] = headers
    _orig_async_init(self, *args, **kwargs)
httpx.AsyncClient.__init__ = _patched_async_init

from dotenv import load_dotenv
load_dotenv()

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from edu.crew import Edu
from edu.scoring import calculate_credibility_score, classify_credibility_level

ORIGINAL_CLAIMS = [
    "Meetings of the WHO FCTC COP are secret and closed",
    "Consumers have no voice at the WHO FCTC COP",
    "Meetings of WHO FCTC COP are biased against harm reduction",
    "Plain packaging does not work, increases illicit trade and reduces tax revenue",
    "The WHO FCTC threatens farmers and ignores livelihoods",
    "The WHO overrides governments and the COP bypasses consensus"
]

claims_text = "\n".join([f"{i+1}. {c}" for i, c in enumerate(ORIGINAL_CLAIMS)])

def parse_report_and_calculate(markdown_text: str):
    """
    Parses component scores from Crew output and recalculates exact scores in Python.
    """
    results = []
    
    # Regex patterns for component scores
    for i, claim in enumerate(ORIGINAL_CLAIMS, 1):
        # Find section for claim i
        # Looks for CLAIM i: or ## CLAIM i
        pattern = rf"(?:###?\s*CLAIM\s*{i}[:\s]+[^\n]*)(.*?)(?=(?:###?\s*CLAIM\s*{i+1}[:\s]+|\Z|##\s*SUMMARY))"
        match = re.search(pattern, markdown_text, re.DOTALL | re.IGNORECASE)
        section = match.group(1) if match else markdown_text
        
        # Extract components Eq, Sc, Ec, Er, Sp
        eq_m = re.search(r"Evidence Quality\s*\([^)]*Eq[^)]*\)\s*:\s*(\d+(?:\.\d+)?)", section, re.IGNORECASE)
        sc_m = re.search(r"Scientific\s*(?:/|&)?\s*(?:Institutional\s*)?Consensus\s*\([^)]*Sc[^)]*\)\s*:\s*(\d+(?:\.\d+)?)", section, re.IGNORECASE)
        ec_m = re.search(r"Evidence Consistency\s*\([^)]*Ec[^)]*\)\s*:\s*(\d+(?:\.\d+)?)", section, re.IGNORECASE)
        er_m = re.search(r"Evidence Recency\s*\([^)]*Er[^)]*\)\s*:\s*(\d+(?:\.\d+)?)", section, re.IGNORECASE)
        sp_m = re.search(r"Scientific\s*(?:/|&)?\s*(?:Governance\s*)?Plausibility\s*\([^)]*Sp[^)]*\)\s*:\s*(\d+(?:\.\d+)?)", section, re.IGNORECASE)
        
        # Fallback to secondary pattern if needed
        if not eq_m:
            eq_m = re.search(r"\bEq\b[^\d]*(\d+(?:\.\d+)?)", section)
        if not sc_m:
            sc_m = re.search(r"\bSc\b[^\d]*(\d+(?:\.\d+)?)", section)
        if not ec_m:
            ec_m = re.search(r"\bEc\b[^\d]*(\d+(?:\.\d+)?)", section)
        if not er_m:
            er_m = re.search(r"\bEr\b[^\d]*(\d+(?:\.\d+)?)", section)
        if not sp_m:
            sp_m = re.search(r"\bSp\b[^\d]*(\d+(?:\.\d+)?)", section)
            
        eq = float(eq_m.group(1)) if eq_m else 0.0
        sc = float(sc_m.group(1)) if sc_m else 0.0
        ec = float(ec_m.group(1)) if ec_m else 0.0
        er = float(er_m.group(1)) if er_m else 0.0
        sp = float(sp_m.group(1)) if sp_m else 0.0
        
        exact_score = calculate_credibility_score(eq, sc, ec, er, sp)
        level = classify_credibility_level(exact_score)
        
        results.append({
            "id": i,
            "claim": claim,
            "components": {
                "Eq": eq,
                "Sc": sc,
                "Ec": ec,
                "Er": er,
                "Sp": sp
            },
            "exact_python_score": exact_score,
            "classification": level
        })
        
    return results

def main():
    print("=" * 80)
    print("RUNNING MULTI-AGENT CREWAI PIPELINE (OPENROUTER: openai/gpt-6-luna)")
    print("Evaluating Original Six WHO FCTC COP Claims with Live Retrieval")
    print("=" * 80)
    print(f"\nExact Claims to Evaluate:\n{claims_text}\n")
    
    inputs = {
        'topic': claims_text.strip(),
        'source_type': 'fctc_cop_policy_misinformation',
        'url': 'https://fctc.who.int',
        'timestamp': '2026-10-29T18:40:00Z'
    }
    
    crew_instance = Edu()
    result = crew_instance.crew().kickoff(inputs=inputs)
    raw_markdown = str(result)
    
    # Save the complete Markdown report without modifying previous reports
    md_filename = "cop_original_six_openrouter_report.md"
    with open(md_filename, "w", encoding="utf-8") as f:
        f.write(raw_markdown)
    print(f"\nReport successfully saved to {md_filename}")
    
    # Parse scores and calculate exact Python scores
    parsed_results = parse_report_and_calculate(raw_markdown)
    
    json_filename = "cop_original_six_openrouter_report.json"
    with open(json_filename, "w", encoding="utf-8") as f:
        json.dump(parsed_results, f, indent=2)
    print(f"Structured JSON data saved to {json_filename}")
    
    print("\n" + "=" * 80)
    print("PARSED EXACT PYTHON RESULTS:")
    print("=" * 80)
    for r in parsed_results:
        comps = r['components']
        print(f"Claim {r['id']}: \"{r['claim']}\"")
        print(f"  Eq={comps['Eq']}, Sc={comps['Sc']}, Ec={comps['Ec']}, Er={comps['Er']}, Sp={comps['Sp']}")
        print(f"  Exact Python Score: {r['exact_python_score']}/100 -> {r['classification']}\n")

if __name__ == '__main__':
    main()
