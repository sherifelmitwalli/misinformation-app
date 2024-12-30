#!/usr/bin/env python
import sys
import warnings
from edu.crew import Edu
import os
from dotenv import load_dotenv
load_dotenv()


# Make sure to install langtrace-python-sdk before running this script
# Run the following command in your terminal:

# Must precede any llm module imports

from langtrace_python_sdk import langtrace
LANGTRACE_API_KEY= os.getenv("LANGTRACE_API_KEY")


langtrace.init(api_key = LANGTRACE_API_KEY)
warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")

# This main file is intended to be a way for you to run your
# crew locally, so refrain from adding unnecessary logic into this file.
# Replace with inputs you want to test with, it will automatically
# interpolate any tasks and agents information


def validate_inputs(inputs: dict) -> bool:
    required_fields = ['topic', 'source_type', 'url', 'timestamp']
    if not all(field in inputs for field in required_fields):
        raise ValueError(f"Missing required fields. Required: {required_fields}")
    
    if not isinstance(inputs['topic'], str) or len(inputs['topic'].strip()) == 0:
        raise ValueError("Topic must be a non-empty string")
    
    valid_source_types = ['news_article', 'scientific_paper', 'social_media', 'website']
    if inputs['source_type'] not in valid_source_types:
        raise ValueError(f"Invalid source_type. Must be one of: {valid_source_types}")
    
    return True

def run():
    """
    Run the misinformation detection crew.
    """
    inputs = {
        'topic': "Smoking reduces life expectancy by 10 years. E-cigarettes are safer than traditional cigarettes. Tobacco companies are not responsible for lung cancer.",
        'source_type': 'news_article',  # Specify the type of source (e.g., news_article, social_media)
        'url': "https://example.com",
        'timestamp': "2024-12-20T12:00:00Z"
    }
    Edu().crew().kickoff(inputs=inputs)


def train():
    """
    Train the crew for a given number of iterations.
    """
    inputs = {
        "topic": "AI LLMs"
    }
    try:
        Edu().crew().train(n_iterations=int(sys.argv[1]), filename=sys.argv[2], inputs=inputs)

    except Exception as e:
        raise Exception(f"An error occurred while training the crew: {e}")

def replay():
    """
    Replay the crew execution from a specific task.
    """
    try:
        Edu().crew().replay(task_id=sys.argv[1])

    except Exception as e:
        raise Exception(f"An error occurred while replaying the crew: {e}")

def test():
    """
    Test the crew execution and returns the results.
    """
    inputs = {
        "topic": "AI LLMs"
    }
    try:
        Edu().crew().test(n_iterations=int(sys.argv[1]), openai_model_name=sys.argv[2], inputs=inputs)

    except Exception as e:
        raise Exception(f"An error occurred while replaying the crew: {e}")
