#!/usr/bin/env python3
"""
RLM - Recursive Language Models
Entry point and demo test
"""

import logging
import random
import string
import sys

from rlm.rlm import RLM
from rlm.config import Config

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def generate_haystack(size: int = 50000, needle: str = None, needle_position: float = 0.5) -> tuple[str, str]:
    """
    Generate a haystack with a hidden needle.

    Args:
        size: Approximate size of the haystack in characters
        needle: The hidden message (auto-generated if None)
        needle_position: Where to place needle (0.0 = start, 1.0 = end)

    Returns:
        Tuple of (haystack_with_needle, needle)
    """
    if needle is None:
        needle = f"SECRET_CODE_{random.randint(1000, 9999)}_MAGIC"

    # Generate random text paragraphs
    words = [
        "lorem", "ipsum", "dolor", "sit", "amet", "consectetur", "adipiscing",
        "elit", "sed", "do", "eiusmod", "tempor", "incididunt", "ut", "labore",
        "et", "dolore", "magna", "aliqua", "enim", "ad", "minim", "veniam",
        "quis", "nostrud", "exercitation", "ullamco", "laboris", "nisi", "aliquip",
        "ex", "ea", "commodo", "consequat", "duis", "aute", "irure", "in",
        "reprehenderit", "voluptate", "velit", "esse", "cillum", "fugiat", "nulla",
        "pariatur", "excepteur", "sint", "occaecat", "cupidatat", "non", "proident",
        "sunt", "culpa", "qui", "officia", "deserunt", "mollit", "anim", "id", "est"
    ]

    paragraphs = []
    current_size = 0

    while current_size < size:
        # Generate a paragraph
        para_length = random.randint(50, 150)
        para_words = [random.choice(words) for _ in range(para_length)]
        para = " ".join(para_words) + ".\n\n"
        paragraphs.append(para)
        current_size += len(para)

    # Join paragraphs
    haystack = "".join(paragraphs)

    # Insert needle at specified position
    insert_pos = int(len(haystack) * needle_position)
    needle_line = f"\n\n[HIDDEN MESSAGE: {needle}]\n\n"
    haystack_with_needle = haystack[:insert_pos] + needle_line + haystack[insert_pos:]

    return haystack_with_needle, needle


def run_needle_test(context_size: int = 50000):
    """
    Run a needle-in-haystack test.

    Args:
        context_size: Size of the haystack in characters
    """
    print(f"\n{'='*60}")
    print(f"RLM Needle-in-Haystack Test")
    print(f"Context size: {context_size:,} characters")
    print(f"{'='*60}\n")

    # Generate test data
    haystack, needle = generate_haystack(size=context_size, needle_position=0.6)
    print(f"Generated haystack with hidden needle: {needle}")
    print(f"Actual haystack size: {len(haystack):,} characters\n")

    # Create RLM instance
    config = Config()
    config.debug = True
    rlm = RLM(config)

    # Run query
    query = "Find the hidden message in the context. What is the SECRET_CODE?"

    print(f"Query: {query}\n")
    print("-" * 60)

    result = rlm.run(query=query, context=haystack)

    print("-" * 60)
    print(f"\nResult:")
    print(f"  Success: {result.success}")
    print(f"  Iterations: {result.iterations}")
    print(f"  Answer: {result.answer}")

    if result.error:
        print(f"  Error: {result.error}")

    # Check if answer contains the needle
    if needle in result.answer:
        print(f"\n✓ SUCCESS: Found the correct needle!")
    else:
        print(f"\n✗ FAILED: Expected to find '{needle}' in answer")

    # Print usage stats
    stats = rlm.get_usage_stats()
    print(f"\nUsage Statistics:")
    print(f"  LLM Calls: {stats['call_count']}")
    print(f"  Total Tokens: {stats['total_tokens']:,}")

    # Print execution history
    if result.execution_history:
        print(f"\nExecution History ({len(result.execution_history)} executions):")
        for i, record in enumerate(result.execution_history[:5]):  # Show first 5
            print(f"  [{i+1}] Iteration {record.iteration}: {'✓' if record.success else '✗'}")
            print(f"      Code: {record.code[:80]}...")
        if len(result.execution_history) > 5:
            print(f"  ... and {len(result.execution_history) - 5} more")

    return result.success


def run_simple_test():
    """Run a simple test without the needle complexity"""
    print(f"\n{'='*60}")
    print(f"RLM Simple Test")
    print(f"{'='*60}\n")

    context = """
    Product Catalog:

    Item 1: Laptop Pro X
    Price: $1299
    Category: Electronics

    Item 2: Wireless Mouse
    Price: $49
    Category: Electronics

    Item 3: Coffee Maker
    Price: $89
    Category: Kitchen

    Item 4: Running Shoes
    Price: $129
    Category: Sports

    Item 5: Desk Lamp
    Price: $45
    Category: Home Office
    """

    query = "What is the total price of all Electronics items?"

    config = Config()
    config.debug = True
    rlm = RLM(config)

    print(f"Query: {query}\n")
    print("-" * 60)

    result = rlm.run(query=query, context=context)

    print("-" * 60)
    print(f"\nResult:")
    print(f"  Success: {result.success}")
    print(f"  Iterations: {result.iterations}")
    print(f"  Answer: {result.answer}")

    # Expected: $1299 + $49 = $1348
    expected = "1348"
    if expected in result.answer:
        print(f"\n✓ SUCCESS: Correct answer!")
    else:
        print(f"\n? Answer verification needed (expected {expected})")

    return result.success


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="RLM - Recursive Language Models")
    parser.add_argument(
        "--test",
        choices=["simple", "needle", "both"],
        default="simple",
        help="Which test to run"
    )
    parser.add_argument(
        "--size",
        type=int,
        default=50000,
        help="Context size for needle test (default: 50000)"
    )

    args = parser.parse_args()

    if args.test == "simple":
        run_simple_test()
    elif args.test == "needle":
        run_needle_test(args.size)
    elif args.test == "both":
        run_simple_test()
        run_needle_test(args.size)
