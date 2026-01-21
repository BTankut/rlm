"""Test for 1M+ character context handling"""

import pytest
import time
from rlm.rlm import RLM
from rlm.config import Config


@pytest.mark.slow
class TestMillionCharContext:
    """Tests for million+ character contexts (LLM-dependent, slow)"""

    @pytest.fixture
    def config_1m(self):
        """Config optimized for 1M context"""
        cfg = Config()
        cfg.debug = False  # Reduce logging overhead
        cfg.max_iterations = 15  # Enough iterations to find needle
        cfg.max_output_chars = 5000
        cfg.enable_sub_calls = True
        return cfg

    @pytest.fixture
    def rlm_1m(self, config_1m):
        return RLM(config_1m)

    def generate_large_context(self, target_size: int, needle: str, position: float = 0.5) -> str:
        """Generate a large context with a hidden needle"""
        # Create repetitive but varied content
        base_paragraphs = [
            "The quick brown fox jumps over the lazy dog. " * 20,
            "Lorem ipsum dolor sit amet, consectetur adipiscing elit. " * 20,
            "In a world of vast data, finding the right information is key. " * 20,
            "Technology continues to advance at an unprecedented rate. " * 20,
            "The importance of efficient algorithms cannot be overstated. " * 20,
        ]

        # Build content
        content_parts = []
        current_size = 0
        para_index = 0

        while current_size < target_size:
            para = base_paragraphs[para_index % len(base_paragraphs)]
            content_parts.append(para + "\n\n")
            current_size += len(para) + 2
            para_index += 1

        content = "".join(content_parts)

        # Insert needle at specified position
        insert_pos = int(len(content) * position)
        needle_marker = f"\n\n[CRITICAL DATA: {needle}]\n\n"

        return content[:insert_pos] + needle_marker + content[insert_pos:]

    @pytest.mark.slow
    def test_1m_context_handling(self, rlm_1m):
        """Test that RLM can handle 1M+ character context"""
        # Generate 1M context
        target_size = 1_000_000
        needle = "SECRET_CODE_12345"

        print(f"\nGenerating {target_size:,} character context...")
        start = time.time()
        context = self.generate_large_context(target_size, needle, position=0.6)
        gen_time = time.time() - start
        print(f"Generated in {gen_time:.2f}s, actual size: {len(context):,} chars")

        assert len(context) >= target_size, f"Context should be >= {target_size} chars"

        # Run query with regex search hint
        query = """Find [CRITICAL DATA: SECRET_CODE_...] in the context using regex.
```repl
import re
match = re.search(r'\\[CRITICAL DATA: (SECRET_CODE_\\d+)\\]', context)
print(f"Found: {match.group(1)}" if match else "Not found - searching chunks...")
```
Report the complete SECRET_CODE value."""

        print("Running RLM query...")
        start = time.time()
        result = rlm_1m.run(query=query, context=context)
        run_time = time.time() - start
        print(f"Completed in {run_time:.2f}s")

        print(f"Success: {result.success}")
        print(f"Iterations: {result.iterations}")
        print(f"Answer: {result.answer[:200] if result.answer else 'None'}")

        # Check if needle was found - in answer OR execution history
        # (local models may not always call FINAL properly)
        found_in_answer = needle in result.answer or "12345" in result.answer
        found_in_history = any(
            needle in (record.output or "")
            for record in result.execution_history
        )

        # STRICT assertion: needle MUST be found in answer OR execution history
        assert found_in_answer or found_in_history, (
            f"1M context test FAILED: Needle '{needle}' not found. "
            f"Answer: {result.answer}, "
            f"History: {[r.output[:100] for r in result.execution_history]}"
        )

    @pytest.mark.slow
    def test_1m_context_stats(self, rlm_1m):
        """Test that 1M context is properly handled and stats are tracked"""
        target_size = 1_000_000
        needle = "TEST_VALUE_99999"

        context = self.generate_large_context(target_size, needle, position=0.3)

        query = f"""Find [CRITICAL DATA: TEST_VALUE_...] in context using regex.
```repl
import re
m = re.search(r'\\[CRITICAL DATA: (TEST_VALUE_\\d+)\\]', context)
print(m.group(1) if m else "NOT_FOUND")
```
Report the value with FINAL()."""

        result = rlm_1m.run(query=query, context=context)

        # Verify stats are tracked (LLM message chars, not full context)
        # The context is stored in REPL state, not sent directly to LLM
        stats = result.usage_stats
        total_chars = stats.get('total_input_chars', 0)
        print(f"Total LLM input chars tracked: {total_chars:,}")

        # Stats should track at least the system prompt and query messages
        assert total_chars > 0, "Stats should track some input chars"
        assert stats.get('root_calls', 0) >= 1, "Should have at least 1 root LLM call"

        # Check if needle was found - in answer OR execution history
        found_in_answer = needle in result.answer or "99999" in result.answer
        found_in_history = any(
            needle in (record.output or "")
            for record in result.execution_history
        )

        assert found_in_answer or found_in_history, (
            f"1M stats test: Needle '{needle}' not found. "
            f"Answer: {result.answer}, History: {[r.output[:100] for r in result.execution_history]}"
        )


if __name__ == "__main__":
    # Quick standalone test
    print("Running 1M context test...")
    cfg = Config()
    cfg.debug = False
    cfg.max_iterations = 5

    rlm = RLM(cfg)
    test = TestMillionCharContext()

    # Generate 1M context
    needle = "QUICK_TEST_777"
    context = test.generate_large_context(1_000_000, needle, 0.5)
    print(f"Context size: {len(context):,} chars")

    query = """Find [CRITICAL DATA: ...] using regex:
```repl
import re
m = re.search(r'\\[CRITICAL DATA: ([^\\]]+)\\]', context)
print(m.group(1) if m else "Not found")
```"""

    result = rlm.run(query=query, context=context)
    print(f"Success: {result.success}")
    print(f"Answer: {result.answer}")
    print(f"Found needle: {needle in str(result.answer)}")
