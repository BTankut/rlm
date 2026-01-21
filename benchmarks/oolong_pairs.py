"""OOLONG-Pairs Benchmark - Paper-aligned (Appendix E.1)

Based on OOLONG trec_coarse split with synthetic pair-finding tasks.
Source: /tmp/arxiv-2512/appendix/sec4-benchmarks.tex

20 tasks that require O(N^2) complexity - finding all pairs of user IDs
that satisfy certain semantic label conditions.

Labels (from TREC question classification):
- description and abstract concept
- entity
- human being
- numeric value
- location
- abbreviation
"""

import random
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional

from rlm import RLM, Config


# All 20 tasks from the paper (sec4-benchmarks.tex lines 16-54)
OOLONG_PAIRS_TASKS = [
    # Task 1
    {
        "id": 1,
        "query": "In the above data, list all pairs of user IDs (no duplicate pairs, list lower ID first) where both users have at least one instance with a numeric value or location.",
        "condition": lambda u1_labels, u2_labels: (
            ("numeric value" in u1_labels or "location" in u1_labels) and
            ("numeric value" in u2_labels or "location" in u2_labels)
        )
    },
    # Task 2
    {
        "id": 2,
        "query": "In the above data, list all pairs of user IDs (no duplicate pairs, list lower ID first) where both users have at least one instance with an entity or human being.",
        "condition": lambda u1_labels, u2_labels: (
            ("entity" in u1_labels or "human being" in u1_labels) and
            ("entity" in u2_labels or "human being" in u2_labels)
        )
    },
    # Task 3
    {
        "id": 3,
        "query": "In the above data, list all pairs of user IDs (no duplicate pairs, list lower ID first) where both users have at least one instance with a description and abstract concept or abbreviation.",
        "condition": lambda u1_labels, u2_labels: (
            ("description and abstract concept" in u1_labels or "abbreviation" in u1_labels) and
            ("description and abstract concept" in u2_labels or "abbreviation" in u2_labels)
        )
    },
    # Task 4
    {
        "id": 4,
        "query": "In the above data, list all pairs of user IDs (no duplicate pairs, list lower ID first) where both users have at least one instance with a human being or location.",
        "condition": lambda u1_labels, u2_labels: (
            ("human being" in u1_labels or "location" in u1_labels) and
            ("human being" in u2_labels or "location" in u2_labels)
        )
    },
    # Task 5
    {
        "id": 5,
        "query": "In the above data, list all pairs of user IDs (no duplicate pairs, list lower ID first) where both users have at least one instance with an entity or numeric value.",
        "condition": lambda u1_labels, u2_labels: (
            ("entity" in u1_labels or "numeric value" in u1_labels) and
            ("entity" in u2_labels or "numeric value" in u2_labels)
        )
    },
    # Task 6
    {
        "id": 6,
        "query": "In the above data, list all pairs of user IDs (no duplicate pairs, list lower ID first) where both users have at least one instance with a location or abbreviation.",
        "condition": lambda u1_labels, u2_labels: (
            ("location" in u1_labels or "abbreviation" in u1_labels) and
            ("location" in u2_labels or "abbreviation" in u2_labels)
        )
    },
    # Task 7
    {
        "id": 7,
        "query": "In the above data, list all pairs of user IDs (no duplicate pairs, list lower ID first) where both users have at least one instance with a description and abstract concept or numeric value.",
        "condition": lambda u1_labels, u2_labels: (
            ("description and abstract concept" in u1_labels or "numeric value" in u1_labels) and
            ("description and abstract concept" in u2_labels or "numeric value" in u2_labels)
        )
    },
    # Task 8
    {
        "id": 8,
        "query": "In the above data, list all pairs of user IDs (no duplicate pairs, list lower ID first) where both users have at least one instance with a human being or description and abstract concept.",
        "condition": lambda u1_labels, u2_labels: (
            ("human being" in u1_labels or "description and abstract concept" in u1_labels) and
            ("human being" in u2_labels or "description and abstract concept" in u2_labels)
        )
    },
    # Task 9
    {
        "id": 9,
        "query": "In the above data, list all pairs of user IDs (no duplicate pairs, list lower ID first) where both users have at least one instance with an entity or location.",
        "condition": lambda u1_labels, u2_labels: (
            ("entity" in u1_labels or "location" in u1_labels) and
            ("entity" in u2_labels or "location" in u2_labels)
        )
    },
    # Task 10
    {
        "id": 10,
        "query": "In the above data, list all pairs of user IDs (no duplicate pairs, list lower ID first) where both users have at least one instance with a numeric value or abbreviation.",
        "condition": lambda u1_labels, u2_labels: (
            ("numeric value" in u1_labels or "abbreviation" in u1_labels) and
            ("numeric value" in u2_labels or "abbreviation" in u2_labels)
        )
    },
    # Task 11
    {
        "id": 11,
        "query": "In the above data, list all pairs of user IDs (no duplicate pairs, list lower ID first) such that one user has at least one instance with entity and one with abbreviation, and the other user has exactly one instance with entity.",
        "condition": lambda u1_labels, u2_labels: (
            (("entity" in u1_labels and "abbreviation" in u1_labels) and u2_labels.count("entity") == 1) or
            (("entity" in u2_labels and "abbreviation" in u2_labels) and u1_labels.count("entity") == 1)
        )
    },
    # Task 12
    {
        "id": 12,
        "query": "In the above data, list all pairs of user IDs (no duplicate pairs, list lower ID first) such that one user has at least two instances with numeric value, and the other user has at least one instance with location and at least one instance with human being.",
        "condition": lambda u1_labels, u2_labels: (
            (u1_labels.count("numeric value") >= 2 and "location" in u2_labels and "human being" in u2_labels) or
            (u2_labels.count("numeric value") >= 2 and "location" in u1_labels and "human being" in u1_labels)
        )
    },
    # Task 13
    {
        "id": 13,
        "query": "In the above data, list all pairs of user IDs (no duplicate pairs, list lower ID first) such that one user has exactly one instance with description and abstract concept, and the other user has at least one instance with abbreviation and at least one instance with entity.",
        "condition": lambda u1_labels, u2_labels: (
            (u1_labels.count("description and abstract concept") == 1 and "abbreviation" in u2_labels and "entity" in u2_labels) or
            (u2_labels.count("description and abstract concept") == 1 and "abbreviation" in u1_labels and "entity" in u1_labels)
        )
    },
    # Task 14
    {
        "id": 14,
        "query": "In the above data, list all pairs of user IDs (no duplicate pairs, list lower ID first) such that one user has at least one instance with human being and at least one instance with numeric value, and the other user has exactly two instances with location.",
        "condition": lambda u1_labels, u2_labels: (
            ("human being" in u1_labels and "numeric value" in u1_labels and u2_labels.count("location") == 2) or
            ("human being" in u2_labels and "numeric value" in u2_labels and u1_labels.count("location") == 2)
        )
    },
    # Task 15
    {
        "id": 15,
        "query": "In the above data, list all pairs of user IDs (no duplicate pairs, list lower ID first) such that one user has at least one instance with entity, at least one instance with location, and at least one instance with abbreviation, and the other user has exactly one instance with numeric value.",
        "condition": lambda u1_labels, u2_labels: (
            ("entity" in u1_labels and "location" in u1_labels and "abbreviation" in u1_labels and u2_labels.count("numeric value") == 1) or
            ("entity" in u2_labels and "location" in u2_labels and "abbreviation" in u2_labels and u1_labels.count("numeric value") == 1)
        )
    },
    # Task 16
    {
        "id": 16,
        "query": "In the above data, list all pairs of user IDs (no duplicate pairs, list lower ID first) such that one user has at least one instance with description and abstract concept and at least one instance with human being, and the other user has at least two instances with entity and exactly one instance with abbreviation.",
        "condition": lambda u1_labels, u2_labels: (
            ("description and abstract concept" in u1_labels and "human being" in u1_labels and u2_labels.count("entity") >= 2 and u2_labels.count("abbreviation") == 1) or
            ("description and abstract concept" in u2_labels and "human being" in u2_labels and u1_labels.count("entity") >= 2 and u1_labels.count("abbreviation") == 1)
        )
    },
    # Task 17
    {
        "id": 17,
        "query": "In the above data, list all pairs of user IDs (no duplicate pairs, list lower ID first) such that one user has exactly one instance with numeric value, and the other user has at least one instance with location and at least one instance with description and abstract concept.",
        "condition": lambda u1_labels, u2_labels: (
            (u1_labels.count("numeric value") == 1 and "location" in u2_labels and "description and abstract concept" in u2_labels) or
            (u2_labels.count("numeric value") == 1 and "location" in u1_labels and "description and abstract concept" in u1_labels)
        )
    },
    # Task 18
    {
        "id": 18,
        "query": "In the above data, list all pairs of user IDs (no duplicate pairs, list lower ID first) such that one user has at least one instance with abbreviation and exactly one instance with human being, and the other user has at least one instance with entity and at least one instance with numeric value.",
        "condition": lambda u1_labels, u2_labels: (
            ("abbreviation" in u1_labels and u1_labels.count("human being") == 1 and "entity" in u2_labels and "numeric value" in u2_labels) or
            ("abbreviation" in u2_labels and u2_labels.count("human being") == 1 and "entity" in u1_labels and "numeric value" in u1_labels)
        )
    },
    # Task 19
    {
        "id": 19,
        "query": "In the above data, list all pairs of user IDs (no duplicate pairs, list lower ID first) such that one user has at least two instances with location and at least one instance with entity, and the other user has exactly one instance with description and abstract concept and exactly one instance with abbreviation.",
        "condition": lambda u1_labels, u2_labels: (
            (u1_labels.count("location") >= 2 and "entity" in u1_labels and u2_labels.count("description and abstract concept") == 1 and u2_labels.count("abbreviation") == 1) or
            (u2_labels.count("location") >= 2 and "entity" in u2_labels and u1_labels.count("description and abstract concept") == 1 and u1_labels.count("abbreviation") == 1)
        )
    },
    # Task 20
    {
        "id": 20,
        "query": "In the above data, list all pairs of user IDs (no duplicate pairs, list lower ID first) such that one user has at least one instance with numeric value and at least one instance with human being, and the other user has at least one instance with location, at least one instance with entity, and exactly one instance with abbreviation.",
        "condition": lambda u1_labels, u2_labels: (
            ("numeric value" in u1_labels and "human being" in u1_labels and "location" in u2_labels and "entity" in u2_labels and u2_labels.count("abbreviation") == 1) or
            ("numeric value" in u2_labels and "human being" in u2_labels and "location" in u1_labels and "entity" in u1_labels and u1_labels.count("abbreviation") == 1)
        )
    },
]

# TREC question classification labels
LABELS = [
    "description and abstract concept",
    "entity",
    "human being",
    "numeric value",
    "location",
    "abbreviation"
]

# Sample questions for each label (TREC-style)
SAMPLE_QUESTIONS = {
    "description and abstract concept": [
        "What is the definition of photosynthesis?",
        "How does gravity work?",
        "What causes earthquakes?",
        "Why is the sky blue?",
        "What is the meaning of democracy?",
    ],
    "entity": [
        "What is the largest planet in our solar system?",
        "What company makes the iPhone?",
        "What is the chemical symbol for gold?",
        "What programming language was created by Guido van Rossum?",
        "What is the capital of France?",
    ],
    "human being": [
        "Who invented the telephone?",
        "Who wrote Romeo and Juliet?",
        "Who was the first person to walk on the moon?",
        "Who painted the Mona Lisa?",
        "Who discovered penicillin?",
    ],
    "numeric value": [
        "How many states are in the USA?",
        "What is the speed of light?",
        "How tall is Mount Everest?",
        "What year did World War II end?",
        "How many bones are in the human body?",
    ],
    "location": [
        "Where is the Eiffel Tower?",
        "Where was pizza invented?",
        "Where do polar bears live?",
        "Where is the Amazon rainforest?",
        "Where is Silicon Valley?",
    ],
    "abbreviation": [
        "What does NASA stand for?",
        "What does CPU mean?",
        "What is the full form of HTML?",
        "What does UNESCO stand for?",
        "What does API mean in programming?",
    ],
}


@dataclass
class OOLONGPairsResult:
    """Result of a single OOLONG-Pairs benchmark run"""
    task_id: int
    context_size: int
    num_users: int
    expected_pairs: list[tuple[int, int]]
    actual_pairs: list[tuple[int, int]]
    precision: float
    recall: float
    f1: float
    success: bool
    iterations: int
    sub_calls: int
    time_seconds: float
    error: Optional[str] = None


def generate_user_data(
    num_users: int,
    instances_per_user: tuple[int, int] = (2, 5),
    seed: int = None
) -> dict:
    """
    Generate synthetic user data with labeled question instances.

    Args:
        num_users: Number of users to generate
        instances_per_user: (min, max) instances per user
        seed: Random seed

    Returns:
        Dict with user_id -> list of (question, label, timestamp)
    """
    if seed is not None:
        random.seed(seed)

    users = {}
    base_date = datetime(2023, 1, 1)

    for user_id in range(1, num_users + 1):
        num_instances = random.randint(*instances_per_user)
        instances = []

        for _ in range(num_instances):
            label = random.choice(LABELS)
            question = random.choice(SAMPLE_QUESTIONS[label])
            # Random date in 2023
            days_offset = random.randint(0, 365)
            timestamp = base_date + timedelta(days=days_offset)

            instances.append({
                "question": question,
                "label": label,
                "timestamp": timestamp.strftime("%Y-%m-%d")
            })

        users[user_id] = instances

    return users


def format_context(users: dict, target_size: int = None) -> str:
    """
    Format user data as context string.

    Args:
        users: User data dict
        target_size: Target context size (pad if needed)

    Returns:
        Formatted context string
    """
    lines = ["USER DATA WITH QUESTION INSTANCES", "=" * 50, ""]
    lines.append("Each user has question instances. Each question can be labeled as one of:")
    lines.append("- description and abstract concept")
    lines.append("- entity")
    lines.append("- human being")
    lines.append("- numeric value")
    lines.append("- location")
    lines.append("- abbreviation")
    lines.append("")
    lines.append("The data does not provide the labels - you need to figure out the label from the semantics of the question.")
    lines.append("")
    lines.append("-" * 50)

    for user_id, instances in sorted(users.items()):
        lines.append(f"\nUser ID: {user_id}")
        for i, inst in enumerate(instances, 1):
            lines.append(f"  Instance {i}: \"{inst['question']}\" (Date: {inst['timestamp']})")
        lines.append("")

    context = "\n".join(lines)

    # Pad to target size if needed
    if target_size and len(context) < target_size:
        # Add filler paragraphs
        filler = "\n\n[Additional system notes: This data represents a sample of user activity. " \
                 "Each question should be classified based on what type of answer it expects. " \
                 "Questions about people should be labeled 'human being'. " \
                 "Questions about numbers, dates, or quantities should be 'numeric value'. " \
                 "Questions about places should be 'location'. " \
                 "Questions about acronyms or initialisms should be 'abbreviation'. " \
                 "Questions about things or organizations should be 'entity'. " \
                 "Questions asking for explanations or definitions should be 'description and abstract concept'.]\n"

        while len(context) < target_size:
            context += filler

    return context[:target_size] if target_size else context


def compute_expected_pairs(users: dict, condition) -> list[tuple[int, int]]:
    """
    Compute expected pairs based on condition function.

    Args:
        users: User data dict
        condition: Lambda function that takes two label lists

    Returns:
        List of (user_id_1, user_id_2) pairs
    """
    pairs = []
    user_ids = sorted(users.keys())

    # Pre-compute labels for each user
    user_labels = {}
    for user_id, instances in users.items():
        user_labels[user_id] = [inst["label"] for inst in instances]

    # Check all pairs
    for i, u1 in enumerate(user_ids):
        for u2 in user_ids[i + 1:]:
            if condition(user_labels[u1], user_labels[u2]):
                pairs.append((u1, u2))

    return pairs


def parse_pairs_from_answer(answer: str) -> list[tuple[int, int]]:
    """
    Parse pairs from LLM answer.

    Expected format: (user_id_1, user_id_2) per line
    """
    import re
    pairs = []

    # Match patterns like (1, 2), (1,2), 1,2, etc.
    pattern = r'\(?\s*(\d+)\s*,\s*(\d+)\s*\)?'
    matches = re.findall(pattern, answer)

    for m in matches:
        u1, u2 = int(m[0]), int(m[1])
        # Ensure lower ID first
        pair = (min(u1, u2), max(u1, u2))
        if pair not in pairs:
            pairs.append(pair)

    return sorted(pairs)


def calculate_metrics(expected: list, actual: list) -> tuple[float, float, float]:
    """Calculate precision, recall, F1"""
    expected_set = set(expected)
    actual_set = set(actual)

    if not actual_set:
        return 0.0, 0.0, 0.0

    true_positives = len(expected_set & actual_set)
    precision = true_positives / len(actual_set) if actual_set else 0.0
    recall = true_positives / len(expected_set) if expected_set else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return precision, recall, f1


def run_oolong_pairs_benchmark(
    config: Config = None,
    task_ids: list[int] = None,
    context_sizes: list[int] = None,
    num_users: int = 10,
    seed: int = 42
) -> list[OOLONGPairsResult]:
    """
    Run the OOLONG-Pairs benchmark.

    Args:
        config: RLM configuration
        task_ids: List of task IDs to run (1-20), default [1]
        context_sizes: Context sizes to test
        num_users: Number of users in data
        seed: Random seed

    Returns:
        List of OOLONGPairsResult objects
    """
    if task_ids is None:
        task_ids = [1]  # Default to task 1
    if context_sizes is None:
        context_sizes = [4096, 8192]

    results = []
    rlm = RLM(config)

    for task_id in task_ids:
        task = OOLONG_PAIRS_TASKS[task_id - 1]

        for size in context_sizes:
            # Generate data
            users = generate_user_data(num_users, seed=seed + task_id + size)
            context = format_context(users, target_size=size)
            expected_pairs = compute_expected_pairs(users, task["condition"])

            # Build query with task instruction
            query = task["query"] + "\n\nIn your answer, list all pairs in the format (user_id_1, user_id_2), separated by newlines."

            print(f"Running OOLONG-Pairs Task {task_id}: {size} chars, {num_users} users, {len(expected_pairs)} expected pairs")

            start_time = time.time()
            try:
                result = rlm.run(query=query, context=context)
                elapsed = time.time() - start_time

                actual_pairs = parse_pairs_from_answer(result.answer)
                precision, recall, f1 = calculate_metrics(expected_pairs, actual_pairs)

                results.append(OOLONGPairsResult(
                    task_id=task_id,
                    context_size=len(context),
                    num_users=num_users,
                    expected_pairs=expected_pairs,
                    actual_pairs=actual_pairs,
                    precision=precision,
                    recall=recall,
                    f1=f1,
                    success=f1 >= 0.5,  # Consider success if F1 >= 50%
                    iterations=result.iterations,
                    sub_calls=result.sub_calls,
                    time_seconds=elapsed,
                    error=result.error
                ))

            except Exception as e:
                elapsed = time.time() - start_time
                results.append(OOLONGPairsResult(
                    task_id=task_id,
                    context_size=len(context),
                    num_users=num_users,
                    expected_pairs=expected_pairs,
                    actual_pairs=[],
                    precision=0.0,
                    recall=0.0,
                    f1=0.0,
                    success=False,
                    iterations=0,
                    sub_calls=0,
                    time_seconds=elapsed,
                    error=str(e)
                ))

    return results


def print_oolong_pairs_report(results: list[OOLONGPairsResult]):
    """Print a summary report of OOLONG-Pairs benchmark results"""
    print("\n" + "=" * 70)
    print("OOLONG-PAIRS BENCHMARK RESULTS")
    print("=" * 70)

    for r in results:
        print(f"\nTask {r.task_id} | Context: {r.context_size:,} chars | Users: {r.num_users}")
        print(f"  Expected pairs: {len(r.expected_pairs)}")
        print(f"  Found pairs:    {len(r.actual_pairs)}")
        print(f"  Precision: {r.precision:.2%} | Recall: {r.recall:.2%} | F1: {r.f1:.2%}")
        print(f"  Success: {'YES' if r.success else 'NO'} | Iterations: {r.iterations} | Sub-calls: {r.sub_calls}")
        print(f"  Time: {r.time_seconds:.2f}s")
        if r.error:
            print(f"  Error: {r.error}")

    # Summary
    total = len(results)
    success_count = sum(1 for r in results if r.success)
    avg_f1 = sum(r.f1 for r in results) / total if total else 0

    print("\n" + "-" * 70)
    print(f"SUMMARY: {success_count}/{total} successful ({100*success_count/total:.1f}%)")
    print(f"Average F1: {avg_f1:.2%}")
    print("=" * 70)


if __name__ == "__main__":
    results = run_oolong_pairs_benchmark(
        task_ids=[1, 2],
        context_sizes=[4096],
        num_users=8
    )
    print_oolong_pairs_report(results)
