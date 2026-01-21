"""Code QA Benchmark - Question answering over code contexts

Tests the model's ability to analyze code and answer questions about it.
"""

import random
import time
from dataclasses import dataclass
from typing import Optional

from rlm import RLM, Config


@dataclass
class CodeQAResult:
    """Result of a single Code QA benchmark run"""
    test_name: str
    code_size: int
    query: str
    expected_answer: str
    actual_answer: str
    success: bool
    iterations: int
    time_seconds: float
    error: Optional[str] = None


# Sample code templates for testing
CODE_TEMPLATES = {
    "function_count": {
        "code": '''
def process_data(data):
    """Process input data"""
    return [x * 2 for x in data]

def validate_input(value):
    """Validate input value"""
    return isinstance(value, (int, float))

def format_output(result):
    """Format the output"""
    return f"Result: {{result}}"

def calculate_sum(numbers):
    """Calculate sum of numbers"""
    return sum(numbers)

def calculate_average(numbers):
    """Calculate average of numbers"""
    if not numbers:
        return 0
    return sum(numbers) / len(numbers)

class DataProcessor:
    def __init__(self):
        self.data = []

    def add(self, item):
        self.data.append(item)

    def process(self):
        return process_data(self.data)
''',
        "query": "How many functions are defined in this code (not counting methods)?",
        "answer": "5"
    },
    "class_attributes": {
        "code": '''
class Configuration:
    """Application configuration"""
    VERSION = "2.5.3"
    DEBUG_MODE = False
    MAX_CONNECTIONS = 100
    TIMEOUT_SECONDS = 30
    API_ENDPOINT = "https://api.example.com"
    RETRY_COUNT = 3

    def __init__(self):
        self.custom_settings = {{}}

    def get_version(self):
        return self.VERSION

    def set_debug(self, enabled):
        self.DEBUG_MODE = enabled
''',
        "query": "What is the value of VERSION in the Configuration class?",
        "answer": "2.5.3"
    },
    "import_count": {
        "code": '''
import os
import sys
import json
import logging
from datetime import datetime, timedelta
from collections import defaultdict, Counter
from typing import Optional, List, Dict

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def main():
    logger = logging.getLogger(__name__)
    logger.info("Starting application")
    data = load_data()
    process(data)

def load_data():
    with open("data.json") as f:
        return json.load(f)

def process(data):
    df = pd.DataFrame(data)
    return df.describe()
''',
        "query": "How many import statements are there in total (counting 'import' and 'from' lines)?",
        "answer": "7"
    },
    "find_constant": {
        "code": '''
# Configuration constants
DATABASE_HOST = "localhost"
DATABASE_PORT = 5432
DATABASE_NAME = "production_db"
DATABASE_USER = "admin"

# API Settings
API_KEY = "sk_live_abc123xyz789"
API_VERSION = "v2"
API_RATE_LIMIT = 1000

# Feature flags
ENABLE_CACHING = True
ENABLE_LOGGING = True
ENABLE_METRICS = False

# Secret value embedded in code
SECRET_TOKEN = "TOKEN_98765_SECURE"

# More settings follow...
LOG_LEVEL = "INFO"
MAX_RETRIES = 5
''',
        "query": "What is the value of SECRET_TOKEN?",
        "answer": "TOKEN_98765_SECURE"
    }
}


def generate_code_qa_test(
    template_name: str = None,
    padding_lines: int = 0
) -> tuple[str, str, str, str]:
    """
    Generate a Code QA test case.

    Args:
        template_name: Specific template to use (random if None)
        padding_lines: Additional comment lines to add for size

    Returns:
        Tuple of (test_name, code, query, expected_answer)
    """
    if template_name is None:
        template_name = random.choice(list(CODE_TEMPLATES.keys()))

    template = CODE_TEMPLATES[template_name]
    code = template["code"]

    # Add padding if requested
    if padding_lines > 0:
        padding = "\n".join([f"# Padding line {i}" for i in range(padding_lines)])
        # Insert padding in the middle
        mid = len(code) // 2
        code = code[:mid] + "\n" + padding + "\n" + code[mid:]

    return template_name, code, template["query"], template["answer"]


def run_code_qa_benchmark(
    config: Config = None,
    templates: list[str] = None,
    padding_sizes: list[int] = None
) -> list[CodeQAResult]:
    """
    Run the Code QA benchmark suite.

    Args:
        config: RLM configuration (uses default if None)
        templates: List of template names to test (all if None)
        padding_sizes: List of padding line counts to test

    Returns:
        List of CodeQAResult objects
    """
    if templates is None:
        templates = list(CODE_TEMPLATES.keys())

    if padding_sizes is None:
        padding_sizes = [0]

    results = []
    rlm = RLM(config)

    for template_name in templates:
        for padding in padding_sizes:
            test_name, code, query, expected = generate_code_qa_test(
                template_name=template_name,
                padding_lines=padding
            )

            print(f"Running Code QA: {test_name} (padding={padding})")

            start_time = time.time()
            try:
                result = rlm.run(query=query, context=code)
                elapsed = time.time() - start_time

                # Check if answer contains expected value
                success = expected in result.answer

                results.append(CodeQAResult(
                    test_name=test_name,
                    code_size=len(code),
                    query=query,
                    expected_answer=expected,
                    actual_answer=result.answer,
                    success=success,
                    iterations=result.iterations,
                    time_seconds=elapsed,
                    error=result.error
                ))

            except Exception as e:
                elapsed = time.time() - start_time
                results.append(CodeQAResult(
                    test_name=test_name,
                    code_size=len(code),
                    query=query,
                    expected_answer=expected,
                    actual_answer="",
                    success=False,
                    iterations=0,
                    time_seconds=elapsed,
                    error=str(e)
                ))

    return results


def print_code_qa_report(results: list[CodeQAResult]):
    """Print a summary report of Code QA benchmark results"""
    print("\n" + "=" * 60)
    print("CODE QA BENCHMARK RESULTS")
    print("=" * 60)

    success_count = sum(1 for r in results if r.success)
    total = len(results)

    print(f"\nOverall: {success_count}/{total} ({100*success_count/total:.1f}%)")

    for r in results:
        status = "✓" if r.success else "✗"
        print(f"\n{status} {r.test_name}")
        print(f"  Code size: {r.code_size} chars")
        print(f"  Expected: {r.expected_answer}")
        print(f"  Got: {r.actual_answer[:50]}{'...' if len(r.actual_answer) > 50 else ''}")
        print(f"  Time: {r.time_seconds:.2f}s, Iterations: {r.iterations}")
        if r.error:
            print(f"  Error: {r.error}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    results = run_code_qa_benchmark()
    print_code_qa_report(results)
