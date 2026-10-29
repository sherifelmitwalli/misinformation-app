#!/usr/bin/env python
"""
Main Entry Point for Tobacco Misinformation Credibility Assessment Pipeline
Implementation corresponding to:
Elmitwalli et al., Frontiers in Artificial Intelligence, 2025.
doi: 10.3389/frai.2025.1659861
"""

import os
import sys
import warnings
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Filter warnings
warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")

# Optional Langtrace observability initialization
LANGTRACE_API_KEY = os.getenv("LANGTRACE_API_KEY")
if LANGTRACE_API_KEY:
    try:
        from langtrace_python_sdk import langtrace
        langtrace.init(api_key=LANGTRACE_API_KEY)
    except Exception as e:
        pass

from edu.crew import Edu

def validate_inputs(inputs: dict) -> bool:
    """Validate required input structure."""
    required_fields = ['topic', 'source_type', 'url', 'timestamp']
    if not all(field in inputs for field in required_fields):
        raise ValueError(f"Missing required fields. Required: {required_fields}")
    
    if not isinstance(inputs['topic'], str) or len(inputs['topic'].strip()) == 0:
        raise ValueError("Topic must be a non-empty string containing the claim(s) to evaluate.")
    
    return True

def run():
    """
    Run the multi-agent misinformation detection crew on sample claims.
    """
    # Default sample claims illustrating diverse categories from the published study
    sample_claims = (
        "1. Smoking reduces life expectancy by 10 years. "
        "2. Plain packaging has no impact on smoking initiation rates. "
        "3. E-cigarettes are completely safe for long-term use."
    )

    inputs = {
        'topic': sample_claims,
        'source_type': 'tobacco_health_claims',
        'url': 'https://www.who.int/health-topics/tobacco',
        'timestamp': '2025-10-01T12:00:00Z'
    }

    validate_inputs(inputs)
    print("=" * 80)
    print("Launching Multi-Agent Credibility Assessment Pipeline (Frontiers in AI, 2025)...")
    print("=" * 80)
    print(f"Input Topic: {inputs['topic']}\n")

    result = Edu().crew().kickoff(inputs=inputs)
    print("\n" + "=" * 80)
    print("PIPELINE ASSESSMENT RESULTS:")
    print("=" * 80)
    print(result)

def train():
    """Train the crew for a given number of iterations."""
    inputs = {"topic": "Tobacco misinformation claims"}
    try:
        Edu().crew().train(n_iterations=int(sys.argv[1]), filename=sys.argv[2], inputs=inputs)
    except Exception as e:
        raise Exception(f"An error occurred while training the crew: {e}")

def replay():
    """Replay the crew execution from a specific task."""
    try:
        Edu().crew().replay(task_id=sys.argv[1])
    except Exception as e:
        raise Exception(f"An error occurred while replaying the crew: {e}")

def test():
    """Test the crew execution and return results."""
    inputs = {"topic": "Tobacco misinformation claims"}
    try:
        Edu().crew().test(n_iterations=int(sys.argv[1]), openai_model_name=sys.argv[2], inputs=inputs)
    except Exception as e:
        raise Exception(f"An error occurred while testing the crew: {e}")

if __name__ == "__main__":
    run()
