"""Tests for OOLONG-Pairs benchmark"""

import pytest
from benchmarks.oolong_pairs import (
    generate_user_data,
    format_context,
    compute_expected_pairs,
    parse_pairs_from_answer,
    calculate_metrics,
    OOLONG_PAIRS_TASKS,
    LABELS,
)


class TestOOLONGPairsDataGeneration:
    """Unit tests for OOLONG-Pairs data generation (fast, no LLM)"""

    def test_generate_user_data(self):
        """Test user data generation"""
        users = generate_user_data(num_users=5, seed=42)

        assert len(users) == 5
        assert all(1 <= uid <= 5 for uid in users.keys())

        # Check each user has instances
        for user_id, instances in users.items():
            assert len(instances) >= 2  # min instances
            for inst in instances:
                assert "question" in inst
                assert "label" in inst
                assert "timestamp" in inst
                assert inst["label"] in LABELS

    def test_generate_user_data_deterministic(self):
        """Test that same seed produces same data"""
        users1 = generate_user_data(num_users=5, seed=123)
        users2 = generate_user_data(num_users=5, seed=123)

        assert users1 == users2

    def test_format_context(self):
        """Test context formatting"""
        users = generate_user_data(num_users=3, seed=42)
        context = format_context(users)

        assert "USER DATA WITH QUESTION INSTANCES" in context
        assert "User ID: 1" in context
        assert "User ID: 2" in context
        assert "User ID: 3" in context

    def test_format_context_with_target_size(self):
        """Test context formatting with target size"""
        users = generate_user_data(num_users=3, seed=42)
        context = format_context(users, target_size=5000)

        # Should be approximately target size
        assert len(context) == 5000

    def test_compute_expected_pairs_task1(self):
        """Test pair computation for task 1 (numeric value OR location)"""
        # Create specific users with known labels
        users = {
            1: [{"question": "q1", "label": "numeric value", "timestamp": "2023-01-01"}],
            2: [{"question": "q2", "label": "location", "timestamp": "2023-01-02"}],
            3: [{"question": "q3", "label": "entity", "timestamp": "2023-01-03"}],
        }

        task = OOLONG_PAIRS_TASKS[0]  # Task 1
        pairs = compute_expected_pairs(users, task["condition"])

        # User 1 (numeric) and User 2 (location) should match
        # User 3 (entity) should not match with anyone
        assert (1, 2) in pairs
        assert (1, 3) not in pairs
        assert (2, 3) not in pairs

    def test_compute_expected_pairs_task2(self):
        """Test pair computation for task 2 (entity OR human being)"""
        users = {
            1: [{"question": "q1", "label": "entity", "timestamp": "2023-01-01"}],
            2: [{"question": "q2", "label": "human being", "timestamp": "2023-01-02"}],
            3: [{"question": "q3", "label": "location", "timestamp": "2023-01-03"}],
        }

        task = OOLONG_PAIRS_TASKS[1]  # Task 2
        pairs = compute_expected_pairs(users, task["condition"])

        assert (1, 2) in pairs
        assert (1, 3) not in pairs
        assert (2, 3) not in pairs


class TestOOLONGPairsParsing:
    """Tests for answer parsing"""

    def test_parse_pairs_simple(self):
        """Test simple pair parsing"""
        answer = "(1, 2)\n(3, 4)\n(5, 6)"
        pairs = parse_pairs_from_answer(answer)

        assert pairs == [(1, 2), (3, 4), (5, 6)]

    def test_parse_pairs_no_parentheses(self):
        """Test parsing without parentheses"""
        answer = "1, 2\n3, 4"
        pairs = parse_pairs_from_answer(answer)

        assert pairs == [(1, 2), (3, 4)]

    def test_parse_pairs_mixed_format(self):
        """Test parsing mixed formats"""
        answer = "Found pairs: (1,2), (3, 4), 5,6"
        pairs = parse_pairs_from_answer(answer)

        assert (1, 2) in pairs
        assert (3, 4) in pairs
        assert (5, 6) in pairs

    def test_parse_pairs_reorders(self):
        """Test that pairs are reordered (lower ID first)"""
        answer = "(5, 2)"
        pairs = parse_pairs_from_answer(answer)

        assert pairs == [(2, 5)]

    def test_parse_pairs_no_duplicates(self):
        """Test that duplicates are removed"""
        answer = "(1, 2)\n(1, 2)\n(2, 1)"  # All same pair
        pairs = parse_pairs_from_answer(answer)

        assert pairs == [(1, 2)]


class TestOOLONGPairsMetrics:
    """Tests for metric calculation"""

    def test_calculate_metrics_perfect(self):
        """Test perfect match"""
        expected = [(1, 2), (3, 4)]
        actual = [(1, 2), (3, 4)]

        precision, recall, f1 = calculate_metrics(expected, actual)

        assert precision == 1.0
        assert recall == 1.0
        assert f1 == 1.0

    def test_calculate_metrics_partial(self):
        """Test partial match"""
        expected = [(1, 2), (3, 4)]
        actual = [(1, 2), (5, 6)]

        precision, recall, f1 = calculate_metrics(expected, actual)

        assert precision == 0.5  # 1 correct out of 2 actual
        assert recall == 0.5     # 1 correct out of 2 expected
        assert f1 == 0.5

    def test_calculate_metrics_no_match(self):
        """Test no match"""
        expected = [(1, 2)]
        actual = [(3, 4)]

        precision, recall, f1 = calculate_metrics(expected, actual)

        assert precision == 0.0
        assert recall == 0.0
        assert f1 == 0.0

    def test_calculate_metrics_empty_actual(self):
        """Test empty actual"""
        expected = [(1, 2)]
        actual = []

        precision, recall, f1 = calculate_metrics(expected, actual)

        assert precision == 0.0
        assert recall == 0.0
        assert f1 == 0.0


class TestOOLONGPairsTasks:
    """Tests for all 20 tasks"""

    def test_all_tasks_defined(self):
        """Test that all 20 tasks are defined"""
        assert len(OOLONG_PAIRS_TASKS) == 20

    def test_all_tasks_have_required_fields(self):
        """Test that all tasks have required fields"""
        for task in OOLONG_PAIRS_TASKS:
            assert "id" in task
            assert "query" in task
            assert "condition" in task
            assert callable(task["condition"])

    def test_task_ids_sequential(self):
        """Test that task IDs are 1-20"""
        task_ids = [t["id"] for t in OOLONG_PAIRS_TASKS]
        assert task_ids == list(range(1, 21))


@pytest.mark.slow
class TestOOLONGPairsIntegration:
    """Integration tests that use real LLM (slow)"""

    @pytest.fixture
    def config(self):
        from rlm import Config
        cfg = Config()
        cfg.debug = True
        cfg.max_iterations = 10
        return cfg

    def test_oolong_pairs_smoke(self, config):
        """Smoke test: run one task with small context"""
        from benchmarks.oolong_pairs import run_oolong_pairs_benchmark, print_oolong_pairs_report

        results = run_oolong_pairs_benchmark(
            config=config,
            task_ids=[1],
            context_sizes=[2048],
            num_users=5,
            seed=42
        )

        assert len(results) == 1
        result = results[0]

        print(f"\nSmoke test result:")
        print(f"  Task: {result.task_id}")
        print(f"  Expected pairs: {result.expected_pairs}")
        print(f"  Actual pairs: {result.actual_pairs}")
        print(f"  F1: {result.f1:.2%}")

        # Just check it ran without error
        assert result.iterations > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
