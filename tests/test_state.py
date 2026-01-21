"""Tests for the state module"""

import pytest
from rlm.core.state import REPLState, ExecutionRecord


class TestREPLState:
    """Tests for REPLState class"""

    def test_initialization(self):
        state = REPLState()
        assert state.globals == {}
        assert state.history == []
        assert state.context_info == {}

    def test_initialize_context(self):
        state = REPLState()
        context = "This is test context" * 100

        state.initialize_context(context, context_type="string")

        assert state.globals['context'] == context
        assert state.context_info['type'] == "string"
        assert state.context_info['total_length'] == len(context)

    def test_get_context_info(self):
        state = REPLState()
        context = "Line 1\nLine 2\nLine 3"
        state.initialize_context(context, context_type="text")

        info = state.get_context_info()

        assert info['context_type'] == "text"
        assert info['context_total_length'] == len(context)
        assert info['context_line_count'] == 3

    def test_get_context_preview_short(self):
        state = REPLState()
        context = "Short context"
        state.initialize_context(context)

        preview = state.get_context_preview(max_chars=500)

        assert preview == context

    def test_get_context_preview_long(self):
        state = REPLState()
        context = "x" * 1000
        state.initialize_context(context)

        preview = state.get_context_preview(max_chars=100)

        assert len(preview) < 200  # Should be truncated
        assert "more characters" in preview

    def test_add_execution_record(self):
        state = REPLState()

        state.add_execution_record(
            iteration=1,
            code="print('hello')",
            output="hello\n",
            success=True
        )

        assert len(state.history) == 1
        record = state.history[0]
        assert record.iteration == 1
        assert record.code == "print('hello')"
        assert record.success is True

    def test_get_variable(self):
        state = REPLState()
        state.globals['test_var'] = 42

        assert state.get_variable('test_var') == 42
        assert state.get_variable('nonexistent') is None

    def test_set_variable(self):
        state = REPLState()
        state.set_variable('my_var', [1, 2, 3])

        assert state.globals['my_var'] == [1, 2, 3]

    def test_get_execution_summary(self):
        state = REPLState()
        state.globals['context'] = "test"
        state.globals['result'] = 42

        state.add_execution_record(1, "code1", "out1", True)
        state.add_execution_record(2, "code2", "out2", False, "error")
        state.add_execution_record(3, "code3", "out3", True)

        summary = state.get_execution_summary()

        assert summary['total_executions'] == 3
        assert summary['successful'] == 2
        assert summary['failed'] == 1
        assert 'context' in summary['variables']
        assert 'result' in summary['variables']

    def test_reset(self):
        state = REPLState()
        state.initialize_context("test context")
        state.add_execution_record(1, "code", "output", True)

        state.reset()

        assert state.globals == {}
        assert state.history == []
        assert state.context_info == {}

    def test_add_llm_query_function(self):
        state = REPLState()

        def mock_llm_query(prompt):
            return f"Response to: {prompt}"

        state.add_llm_query_function(mock_llm_query)

        assert 'llm_query' in state.globals
        assert state.globals['llm_query']("test") == "Response to: test"


class TestExecutionRecord:
    """Tests for ExecutionRecord dataclass"""

    def test_creation(self):
        record = ExecutionRecord(
            iteration=1,
            code="print('test')",
            output="test\n",
            success=True,
            error=None
        )

        assert record.iteration == 1
        assert record.code == "print('test')"
        assert record.output == "test\n"
        assert record.success is True
        assert record.error is None
        assert record.timestamp is not None

    def test_creation_with_error(self):
        record = ExecutionRecord(
            iteration=2,
            code="x = 1/0",
            output="",
            success=False,
            error="ZeroDivisionError: division by zero"
        )

        assert record.success is False
        assert "ZeroDivisionError" in record.error
