"""OOLONG Benchmark - Long-context question answering with sub-calls

Based on the OOLONG paper methodology for testing recursive LLM architectures.
Tests the model's ability to use llm_query() for processing large contexts.
"""

import random
import time
from dataclasses import dataclass
from typing import Optional

from rlm import RLM, Config


@dataclass
class OOLONGResult:
    """Result of a single OOLONG benchmark run"""
    context_size: int
    query: str
    expected_answer: str
    actual_answer: str
    success: bool
    iterations: int
    sub_calls: int
    time_seconds: float
    error: Optional[str] = None


def generate_oolong_test(
    context_size: int,
    num_documents: int = 10,
    seed: Optional[int] = None
) -> tuple[str, str, str]:
    """
    Generate an OOLONG-style test case.

    Creates multiple "documents" with facts, then asks a question
    that requires finding and aggregating information.

    Args:
        context_size: Approximate total character count
        num_documents: Number of documents to generate
        seed: Random seed for reproducibility

    Returns:
        Tuple of (context, query, expected_answer)
    """
    if seed is not None:
        random.seed(seed)

    # Generate document topics and facts
    topics = [
        "Project Alpha", "Initiative Beta", "Program Gamma", "Task Delta",
        "Operation Epsilon", "Mission Zeta", "Plan Eta", "Strategy Theta",
        "Campaign Iota", "Effort Kappa"
    ]

    # Filler words for padding
    filler_words = [
        "lorem", "ipsum", "dolor", "sit", "amet", "consectetur",
        "adipiscing", "elit", "sed", "do", "eiusmod", "tempor"
    ]

    chars_per_doc = context_size // num_documents
    documents = []

    # Target fact - will be the answer
    target_doc_idx = random.randint(0, num_documents - 1)
    target_value = f"VALUE_{random.randint(10000, 99999)}"
    target_topic = topics[target_doc_idx % len(topics)]

    for i in range(num_documents):
        topic = topics[i % len(topics)]

        # Create document header
        doc = f"\n=== DOCUMENT {i + 1}: {topic} ===\n\n"

        # Add filler content
        filler_chars = chars_per_doc - len(doc) - 100
        while len(doc) < chars_per_doc - 100:
            paragraph = " ".join(random.choices(filler_words, k=50)) + ".\n\n"
            doc += paragraph

        # Add the key fact
        if i == target_doc_idx:
            doc += f"\nKEY FINDING: The critical identifier for {topic} is {target_value}.\n"
        else:
            doc += f"\nKEY FINDING: The status of {topic} is ACTIVE.\n"

        # More filler
        doc += " ".join(random.choices(filler_words, k=30)) + ".\n"

        documents.append(doc)

    context = "\n".join(documents)
    query = f"What is the critical identifier for {target_topic}? Search through all documents to find the exact value."

    return context, query, target_value


def run_oolong_benchmark(
    config: Config = None,
    context_sizes: list[int] = None,
    num_trials: int = 1,
    seed: int = 42
) -> list[OOLONGResult]:
    """
    Run the OOLONG benchmark suite.

    Args:
        config: RLM configuration (uses default if None)
        context_sizes: List of context sizes to test
        num_trials: Number of trials per size
        seed: Base random seed

    Returns:
        List of OOLONGResult objects
    """
    if context_sizes is None:
        context_sizes = [10000, 25000, 50000, 100000]

    results = []
    rlm = RLM(config)

    for size in context_sizes:
        for trial in range(num_trials):
            trial_seed = seed + size + trial

            context, query, expected = generate_oolong_test(
                context_size=size,
                num_documents=max(5, size // 10000),
                seed=trial_seed
            )

            print(f"Running OOLONG: {size} chars, trial {trial + 1}/{num_trials}")

            start_time = time.time()
            try:
                result = rlm.run(query=query, context=context)
                elapsed = time.time() - start_time

                success = expected in result.answer
                results.append(OOLONGResult(
                    context_size=size,
                    query=query,
                    expected_answer=expected,
                    actual_answer=result.answer,
                    success=success,
                    iterations=result.iterations,
                    sub_calls=result.sub_calls,
                    time_seconds=elapsed,
                    error=result.error
                ))

            except Exception as e:
                elapsed = time.time() - start_time
                results.append(OOLONGResult(
                    context_size=size,
                    query=query,
                    expected_answer=expected,
                    actual_answer="",
                    success=False,
                    iterations=0,
                    sub_calls=0,
                    time_seconds=elapsed,
                    error=str(e)
                ))

    return results


def print_oolong_report(results: list[OOLONGResult]):
    """Print a summary report of OOLONG benchmark results"""
    print("\n" + "=" * 60)
    print("OOLONG BENCHMARK RESULTS")
    print("=" * 60)

    # Group by context size
    by_size = {}
    for r in results:
        if r.context_size not in by_size:
            by_size[r.context_size] = []
        by_size[r.context_size].append(r)

    for size in sorted(by_size.keys()):
        size_results = by_size[size]
        success_count = sum(1 for r in size_results if r.success)
        avg_time = sum(r.time_seconds for r in size_results) / len(size_results)
        avg_iterations = sum(r.iterations for r in size_results) / len(size_results)
        avg_subcalls = sum(r.sub_calls for r in size_results) / len(size_results)

        print(f"\nContext size: {size:,} chars")
        print(f"  Success rate: {success_count}/{len(size_results)} ({100*success_count/len(size_results):.1f}%)")
        print(f"  Avg time: {avg_time:.2f}s")
        print(f"  Avg iterations: {avg_iterations:.1f}")
        print(f"  Avg sub-calls: {avg_subcalls:.1f}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    results = run_oolong_benchmark(
        context_sizes=[10000, 25000],
        num_trials=2
    )
    print_oolong_report(results)
