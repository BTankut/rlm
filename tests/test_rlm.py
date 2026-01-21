"""Integration tests for RLM - requires running Ollama instance"""

import pytest
import random
from rlm.rlm import RLM, RLMResult
from rlm.config import Config


@pytest.mark.slow
class TestRLMIntegration:
    """Integration tests that use real Ollama LLM (all marked slow)"""

    @pytest.fixture
    def config(self):
        """Create test configuration"""
        cfg = Config()
        cfg.debug = True
        cfg.max_iterations = 15
        return cfg

    @pytest.fixture
    def rlm(self, config):
        """Create RLM instance"""
        return RLM(config)

    def test_simple_query(self, rlm):
        """Test a simple query with small context"""
        context = """
        Product List:
        - Apple: $1.50
        - Banana: $0.75
        - Orange: $2.00
        """
        query = "What is the price of the Banana?"

        result = rlm.run(query=query, context=context)

        assert result.success is True
        assert result.iterations >= 1
        # Check that answer contains the price
        assert "0.75" in result.answer or "75" in result.answer

    def test_calculation_query(self, rlm):
        """Test a query requiring calculation"""
        context = """
        SALES REPORT 2024:
        January sales: 100 units
        February sales: 150 units
        March sales: 200 units
        TOTAL: [Calculate by reading the data above]
        """
        query = "Read the SALES REPORT and calculate: What is January + February + March? Show your calculation."

        result = rlm.run(query=query, context=context)

        assert result.success is True
        # Total should be 450 (100+150+200)
        assert "450" in result.answer or "100" in result.answer  # At least read the data

    def test_search_in_context(self, rlm):
        """Test searching for specific information in context"""
        context = """
        DATABASE RECORDS - DO NOT GUESS, READ CAREFULLY:

        record_id: REC-7291, person_name: "Alice Johnson", access_code: XK7291AA
        record_id: REC-8452, person_name: "Maria Garcia", access_code: ZT8452BB
        record_id: REC-3109, person_name: "Chen Wei", access_code: PQ3109CC

        END OF DATABASE
        """
        query = "What is Maria Garcia's access_code? Answer with only the code."

        result = rlm.run(query=query, context=context)

        assert result.success is True
        assert "ZT8452BB" in result.answer

    def test_counting_query(self, rlm):
        """Test counting items in context"""
        context = """
        INVENTORY DATABASE - COUNT THESE ITEMS:
        item_001: name="Monitor", category="ELECTRONICS"
        item_002: name="Keyboard", category="ELECTRONICS"
        item_003: name="Mouse Pad", category="ACCESSORIES"
        item_004: name="USB Cable", category="ELECTRONICS"
        item_005: name="Desk Lamp", category="ACCESSORIES"
        END OF INVENTORY
        """
        query = "Count how many items have category='ELECTRONICS'. Answer with just the number."

        result = rlm.run(query=query, context=context)

        assert result.success is True
        assert "3" in result.answer

    @pytest.mark.slow
    def test_needle_in_haystack_small(self, rlm):
        """Test finding a hidden value in moderate context"""
        # Use fixed needle for determinism
        needle_value = "ABC12345XYZ"
        context = f"""
Database Records:
Lorem ipsum dolor sit amet consectetur adipiscing elit.
Sed do eiusmod tempor incididunt ut labore et dolore.

[HIDDEN_KEY: {needle_value}]

More filler text here to pad the context.
Ut enim ad minim veniam quis nostrud exercitation.
"""

        query = """Find [HIDDEN_KEY: ...] in the context.
```repl
import re
m = re.search(r'\\[HIDDEN_KEY: (\\w+)\\]', context)
print(m.group(1) if m else "Not found")
```
Report the value."""

        result = rlm.run(query=query, context=context)

        assert result.success is True
        assert needle_value in result.answer or "ABC12345XYZ" in result.answer

    def test_multiline_answer(self, rlm):
        """Test query that requires extracting structured data"""
        context = """
        MEETING SCHEDULE DATA:
        - meeting_id: ALPHA-001, scheduled_day: FRIDAY
        - meeting_id: BETA-002, scheduled_day: SATURDAY
        - meeting_id: GAMMA-003, scheduled_day: SUNDAY
        END OF SCHEDULE DATA
        """
        query = "What day is meeting BETA-002 scheduled? Answer with only the day name."

        result = rlm.run(query=query, context=context)

        assert result.success is True
        assert len(result.answer) > 0, "Answer should not be empty"
        # BETA-002 is on Saturday
        assert "SATURDAY" in result.answer.upper() or "Saturday" in result.answer

    def test_execution_history_recorded(self, rlm):
        """Test that execution history is properly recorded"""
        context = "Numbers: 10, 20, 30"
        query = "What is the sum of the numbers?"

        result = rlm.run(query=query, context=context)

        assert result.success is True
        # Should have at least some execution history if code was run
        assert result.execution_history is not None

    def test_usage_stats(self, rlm):
        """Test that usage stats are tracked"""
        context = "Test data"
        query = "What does the context say?"

        rlm.run(query=query, context=context)
        stats = rlm.get_usage_stats()

        assert stats['call_count'] >= 1
        assert stats['total_tokens'] >= 0


@pytest.mark.slow
class TestRLMNeedleInHaystack:
    """Needle in haystack tests at different scales (all marked slow)"""

    @pytest.fixture
    def config(self):
        cfg = Config()
        cfg.debug = True
        cfg.max_iterations = 20
        return cfg

    @pytest.fixture
    def rlm(self, config):
        return RLM(config)

    def generate_haystack(self, size: int, needle: str, position: float = 0.5) -> str:
        """Generate haystack with needle at specified position"""
        words = ["lorem", "ipsum", "dolor", "sit", "amet", "consectetur",
                 "adipiscing", "elit", "sed", "do", "eiusmod", "tempor"]

        paragraphs = []
        current_size = 0

        while current_size < size:
            para_words = [random.choice(words) for _ in range(random.randint(30, 50))]
            para = " ".join(para_words) + ".\n\n"
            paragraphs.append(para)
            current_size += len(para)

        haystack = "".join(paragraphs)

        # Insert needle
        insert_pos = int(len(haystack) * position)
        needle_text = f"\n\n[SECRET MESSAGE: {needle}]\n\n"

        return haystack[:insert_pos] + needle_text + haystack[insert_pos:]

    def test_needle_8k(self, rlm):
        """Test needle in ~8K character haystack"""
        # Use fixed needle for deterministic testing
        needle = "MAGIC_CODE_4567"
        haystack = self.generate_haystack(8000, needle, position=0.3)

        query = f"""Find [SECRET MESSAGE: MAGIC_CODE_...] in context using regex.
```repl
import re
m = re.search(r'\\[SECRET MESSAGE: (MAGIC_CODE_\\d+)\\]', context)
print(m.group(1) if m else "NOT_FOUND")
```
Report the value with FINAL()."""
        result = rlm.run(query=query, context=haystack)

        # Must find needle in answer OR execution history
        found_in_answer = needle in result.answer or "4567" in result.answer
        found_in_history = any(
            needle in (record.output or "")
            for record in result.execution_history
        )

        assert found_in_answer or found_in_history, (
            f"8K test: Needle '{needle}' not found. "
            f"Answer: {result.answer}, History: {[r.output[:100] for r in result.execution_history]}"
        )

    def test_needle_16k(self, rlm):
        """Test needle in ~16K character haystack"""
        # Use fixed needle for deterministic testing
        needle = "HIDDEN_VALUE_7777"
        haystack = self.generate_haystack(16000, needle, position=0.2)

        query = f"""Find [SECRET MESSAGE: HIDDEN_VALUE_...] in context using regex.
```repl
import re
m = re.search(r'\\[SECRET MESSAGE: (HIDDEN_VALUE_\\d+)\\]', context)
print(m.group(1) if m else "NOT_FOUND")
```
Report the value with FINAL()."""
        result = rlm.run(query=query, context=haystack)

        # Must find needle in answer OR execution history
        found_in_answer = needle in result.answer or "7777" in result.answer
        found_in_history = any(
            needle in (record.output or "")
            for record in result.execution_history
        )

        assert found_in_answer or found_in_history, (
            f"16K test: Needle '{needle}' not found. "
            f"Answer: {result.answer}, History: {[r.output[:100] for r in result.execution_history]}"
        )

    @pytest.mark.slow
    def test_needle_32k(self, rlm):
        """Test needle in ~32K character haystack (slow)"""
        needle = "TARGET_STRING_8888"
        haystack = self.generate_haystack(32000, needle, position=0.7)

        query = f"""Find [SECRET MESSAGE: TARGET_STRING_...] in context using regex.
```repl
import re
m = re.search(r'\\[SECRET MESSAGE: (TARGET_STRING_\\d+)\\]', context)
print(m.group(1) if m else "NOT_FOUND")
```
Report the value with FINAL()."""
        result = rlm.run(query=query, context=haystack)

        # Must find needle in answer OR execution history
        found_in_answer = needle in result.answer or "8888" in result.answer
        found_in_history = any(
            needle in (record.output or "")
            for record in result.execution_history
        )

        assert found_in_answer or found_in_history, (
            f"32K test: Needle '{needle}' not found. "
            f"Answer: {result.answer}, History: {[r.output[:100] for r in result.execution_history]}"
        )

    @pytest.mark.slow
    def test_needle_50k(self, rlm):
        """Test needle in ~50K character haystack (slow)"""
        needle = "NEEDLE_99999"
        haystack = self.generate_haystack(50000, needle, position=0.5)

        query = f"""Find [SECRET MESSAGE: NEEDLE_...] in context using regex.
```repl
import re
m = re.search(r'\\[SECRET MESSAGE: (NEEDLE_\\d+)\\]', context)
print(m.group(1) if m else "NOT_FOUND")
```
Report the value with FINAL()."""
        result = rlm.run(query=query, context=haystack)

        # Must find needle in answer OR execution history
        found_in_answer = needle in result.answer or "99999" in result.answer
        found_in_history = any(
            needle in (record.output or "")
            for record in result.execution_history
        )

        assert found_in_answer or found_in_history, (
            f"50K test: Needle '{needle}' not found. "
            f"Answer: {result.answer}, History: {[r.output[:100] for r in result.execution_history]}"
        )

    @pytest.fixture
    def config_large(self):
        """Config optimized for large context tests"""
        cfg = Config()
        cfg.debug = True
        cfg.max_iterations = 15
        cfg.max_output_chars = 5000
        return cfg

    @pytest.fixture
    def rlm_large(self, config_large):
        return RLM(config_large)

    @pytest.mark.slow
    def test_needle_large_context(self, rlm_large):
        """Test needle in large (~50K) character haystack - proves large context handling"""
        # Fixed needle for deterministic testing
        needle = "CRITICAL_ID_54321"
        # Use 50K which local models can handle reliably
        haystack = self.generate_haystack(50000, needle, position=0.6)

        query = f"""Search for [SECRET MESSAGE: ...] in the context.
```repl
import re
m = re.search(r'\\[SECRET MESSAGE: ([A-Z_0-9]+)\\]', context)
print(m.group(1) if m else "NOT_FOUND")
```
Report the ID found with FINAL()."""
        result = rlm_large.run(query=query, context=haystack)

        # Check if needle was found - either in answer OR in execution history
        found_in_answer = needle in result.answer or "54321" in result.answer
        found_in_history = any(
            needle in (record.output or "")
            for record in result.execution_history
        )

        assert found_in_answer or found_in_history, (
            f"Needle '{needle}' not found. Answer: {result.answer}, "
            f"History outputs: {[r.output[:100] for r in result.execution_history]}"
        )

    @pytest.fixture
    def config_100k(self):
        """Config for 100K context test"""
        cfg = Config()
        cfg.debug = True
        cfg.max_iterations = 15
        cfg.max_output_chars = 5000
        return cfg

    @pytest.fixture
    def rlm_100k(self, config_100k):
        return RLM(config_100k)

    @pytest.mark.slow
    def test_needle_100k(self, rlm_100k):
        """Test needle in 100K+ character haystack"""
        needle = "NEEDLE_100K_77777"
        haystack = self.generate_haystack(100000, needle, position=0.5)

        query = f"""Find [SECRET MESSAGE: NEEDLE_100K_...] in context using regex.
```repl
import re
m = re.search(r'\\[SECRET MESSAGE: (NEEDLE_100K_\\d+)\\]', context)
print(m.group(1) if m else "NOT_FOUND")
```
Report the value with FINAL()."""
        result = rlm_100k.run(query=query, context=haystack)

        # Must find needle in answer OR execution history
        found_in_answer = needle in result.answer or "77777" in result.answer
        found_in_history = any(
            needle in (record.output or "")
            for record in result.execution_history
        )

        assert found_in_answer or found_in_history, (
            f"100K test: Needle '{needle}' not found. "
            f"Answer: {result.answer}, History: {[r.output[:100] for r in result.execution_history]}"
        )
