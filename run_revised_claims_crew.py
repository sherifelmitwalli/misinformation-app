import os
import sys

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

claims_text = """
1. Meetings of WHO FCTC conference of the parties are biased against harm reduction.
2. The WHO FCTC Conference of the Parties exhibits institutional bias against tobacco harm reduction.
3. The WHO FCTC treaty excludes harm reduction from tobacco control strategies.
4. The WHO FCTC COP ignores evidence on e-cigarettes and blocks smokers from quitting.
5. The WHO FCTC adopts a precautionary approach toward novel nicotine products due to youth addiction risks and evolving toxicology.
6. Tobacco companies use harm reduction rhetoric to lobby against FCTC regulations and promote novel nicotine products to youth.
"""

def main():
    print("================================================================================")
    print("RUNNING MULTI-AGENT PIPELINE ON REVISED HARM REDUCTION PHRASINGS & FRAMINGS")
    print("================================================================================\n")
    
    inputs = {
        'topic': claims_text.strip(),
        'source_type': 'tobacco_harm_reduction_claims',
        'url': 'https://fctc.who.int',
        'timestamp': '2026-10-29T18:25:00Z'
    }
    
    crew_instance = Edu()
    result = crew_instance.crew().kickoff(inputs=inputs)
    
    print("\n================== PIPELINE EXECUTION COMPLETE ==================")
    output_str = str(result)
    with open("revised_claims_crew_report.md", "w", encoding="utf-8") as f:
        f.write(output_str)
    print("Done! Report saved to revised_claims_crew_report.md")

if __name__ == '__main__':
    main()
