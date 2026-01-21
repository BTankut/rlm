"""
S-NIAH (Single Needle in a Haystack) Benchmark

Tests the model's ability to find a single piece of information
hidden in a large context of varying sizes.
"""

import random
import time
from typing import Optional
from dataclasses import dataclass

from rlm.rlm import RLM, RLMResult
from rlm.config import Config


@dataclass
class BenchmarkResult:
    """Result of a single benchmark run"""
    context_size: int
    needle: str
    found: bool
    answer: str
    iterations: int
    sub_calls: int
    elapsed_time: float
    needle_position: float


def generate_haystack(size: int) -> str:
    """Generate random filler text of approximately the given size"""
    words = [
        "lorem", "ipsum", "dolor", "sit", "amet", "consectetur", "adipiscing",
        "elit", "sed", "do", "eiusmod", "tempor", "incididunt", "ut", "labore",
        "et", "dolore", "magna", "aliqua", "enim", "ad", "minim", "veniam",
        "quis", "nostrud", "exercitation", "ullamco", "laboris", "nisi", "aliquip",
        "ex", "ea", "commodo", "consequat", "duis", "aute", "irure", "in",
        "reprehenderit", "voluptate", "velit", "esse", "cillum", "fugiat", "nulla"
    ]

    paragraphs = []
    current_size = 0

    while current_size < size:
        para_length = random.randint(40, 80)
        para_words = [random.choice(words) for _ in range(para_length)]
        para = " ".join(para_words) + ".\n\n"
        paragraphs.append(para)
        current_size += len(para)

    return "".join(paragraphs)[:size]


def generate_s_niah_test(
    size: int,
    needle: str = None,
    position: float = None
) -> tuple[str, str, float]:
    """
    Generate a single needle-in-haystack test case.

    Args:
        size: Target size of context in characters
        needle: The needle to hide (auto-generated if None)
        position: Where to place needle (0.0-1.0, random if None)

    Returns:
        Tuple of (context_with_needle, needle, actual_position)
    """
    if needle is None:
        needle = f"SECRET_VALUE_{random.randint(10000, 99999)}"

    if position is None:
        position = random.uniform(0.1, 0.9)

    haystack = generate_haystack(size)

    # Insert needle at position
    insert_idx = int(len(haystack) * position)
    needle_text = f"\n\n[IMPORTANT: The secret value is {needle}]\n\n"

    context = haystack[:insert_idx] + needle_text + haystack[insert_idx:]

    return context, needle, position


def run_single_test(
    rlm: RLM,
    size: int,
    needle: str = None,
    position: float = None
) -> BenchmarkResult:
    """Run a single S-NIAH test"""
    context, needle, actual_position = generate_s_niah_test(size, needle, position)

    query = f"""Find the secret value in the context.
    Look for text marked as [IMPORTANT: ...].
    Search through the ENTIRE context systematically.
    Report ONLY the secret value you find."""

    start_time = time.time()
    result = rlm.run(query=query, context=context)
    elapsed = time.time() - start_time

    found = needle in result.answer or needle.split("_")[-1] in result.answer

    return BenchmarkResult(
        context_size=len(context),
        needle=needle,
        found=found,
        answer=result.answer,
        iterations=result.iterations,
        sub_calls=result.sub_calls,
        elapsed_time=elapsed,
        needle_position=actual_position
    )


def run_s_niah_benchmark(
    sizes: list[int] = None,
    num_trials: int = 3,
    config: Config = None
) -> dict:
    """
    Run the S-NIAH benchmark suite.

    Args:
        sizes: List of context sizes to test (default: [8K, 16K, 32K, 64K])
        num_trials: Number of trials per size
        config: RLM configuration to use

    Returns:
        Dict mapping size to accuracy
    """
    if sizes is None:
        sizes = [8000, 16000, 32000, 64000]

    if config is None:
        config = Config()
        config.max_iterations = 25

    rlm = RLM(config)
    results = {}

    for size in sizes:
        print(f"\nTesting {size:,} characters...")
        successes = 0

        for trial in range(num_trials):
            result = run_single_test(rlm, size)

            status = "✓" if result.found else "✗"
            print(f"  Trial {trial + 1}: {status} "
                  f"(pos={result.needle_position:.1%}, "
                  f"iter={result.iterations}, "
                  f"time={result.elapsed_time:.1f}s)")

            if result.found:
                successes += 1

        accuracy = successes / num_trials
        results[size] = accuracy
        print(f"  Accuracy: {accuracy:.1%}")

    return results


def run_positional_benchmark(
    size: int = 32000,
    positions: list[float] = None,
    num_trials: int = 3,
    config: Config = None
) -> dict:
    """
    Test accuracy at different needle positions.

    Args:
        size: Context size to test
        positions: List of positions (0.0-1.0) to test
        num_trials: Number of trials per position
        config: RLM configuration

    Returns:
        Dict mapping position to accuracy
    """
    if positions is None:
        positions = [0.1, 0.25, 0.5, 0.75, 0.9]

    if config is None:
        config = Config()
        config.max_iterations = 25

    rlm = RLM(config)
    results = {}

    for pos in positions:
        print(f"\nTesting position {pos:.0%}...")
        successes = 0

        for trial in range(num_trials):
            result = run_single_test(rlm, size, position=pos)
            if result.found:
                successes += 1

        accuracy = successes / num_trials
        results[pos] = accuracy
        print(f"  Accuracy: {accuracy:.1%}")

    return results


if __name__ == "__main__":
    print("RLM S-NIAH Benchmark")
    print("=" * 50)

    results = run_s_niah_benchmark(
        sizes=[8000, 16000, 32000],
        num_trials=3
    )

    print("\n" + "=" * 50)
    print("Final Results:")
    for size, accuracy in results.items():
        print(f"  {size:,} chars: {accuracy:.1%}")
