"""Tests for the executor module"""

import pytest
from rlm.core.executor import execute_code, ExecutionResult


class TestExecuteCode:
    """Tests for execute_code function"""

    def test_simple_print(self):
        globals_dict = {}
        result = execute_code('print("hello")', globals_dict)

        assert result.success is True
        assert "hello" in result.stdout
        assert result.error is None

    def test_variable_assignment(self):
        globals_dict = {}
        result = execute_code('x = 42', globals_dict)

        assert result.success is True
        assert globals_dict.get('x') == 42

    def test_variable_persistence(self):
        globals_dict = {}

        # First execution
        execute_code('x = 10', globals_dict)

        # Second execution uses x
        result = execute_code('y = x * 2\nprint(y)', globals_dict)

        assert result.success is True
        assert globals_dict.get('y') == 20
        assert "20" in result.stdout

    def test_syntax_error(self):
        globals_dict = {}
        result = execute_code('if True print("bad")', globals_dict)

        assert result.success is False
        assert result.error is not None
        assert "SyntaxError" in result.error

    def test_runtime_error(self):
        globals_dict = {}
        result = execute_code('x = 1 / 0', globals_dict)

        assert result.success is False
        assert result.error is not None
        assert "ZeroDivisionError" in result.error

    def test_name_error(self):
        globals_dict = {}
        result = execute_code('print(undefined_variable)', globals_dict)

        assert result.success is False
        assert result.error is not None
        assert "NameError" in result.error

    def test_multiline_code(self):
        globals_dict = {}
        code = """
result = []
for i in range(5):
    result.append(i * 2)
print(result)
"""
        result = execute_code(code, globals_dict)

        assert result.success is True
        assert "[0, 2, 4, 6, 8]" in result.stdout

    def test_context_variable(self):
        """Test that context variable is accessible"""
        globals_dict = {'context': 'This is the context data'}
        code = 'print(len(context))'
        result = execute_code(code, globals_dict)

        assert result.success is True
        assert "24" in result.stdout  # len("This is the context data")

    def test_import_works(self):
        globals_dict = {}
        code = """
import re
matches = re.findall(r'\\d+', 'abc123def456')
print(matches)
"""
        result = execute_code(code, globals_dict)

        assert result.success is True
        assert "123" in result.stdout
        assert "456" in result.stdout

    def test_partial_output_on_error(self):
        """Test that output before error is captured"""
        globals_dict = {}
        code = """
print("before error")
x = 1 / 0
print("after error")
"""
        result = execute_code(code, globals_dict)

        assert result.success is False
        assert "before error" in result.stdout
        assert "after error" not in result.stdout

    def test_list_comprehension(self):
        globals_dict = {}
        code = """
numbers = [x**2 for x in range(10) if x % 2 == 0]
print(numbers)
"""
        result = execute_code(code, globals_dict)

        assert result.success is True
        assert "[0, 4, 16, 36, 64]" in result.stdout

    def test_function_definition(self):
        globals_dict = {}

        # Define function
        execute_code("""
def greet(name):
    return f"Hello, {name}!"
""", globals_dict)

        # Use function
        result = execute_code('print(greet("World"))', globals_dict)

        assert result.success is True
        assert "Hello, World!" in result.stdout

    def test_string_search_in_context(self):
        """Test typical RLM use case: searching in context"""
        context = "The secret code is XYZ789. Don't tell anyone."
        globals_dict = {'context': context}

        code = """
import re
match = re.search(r'secret code is (\\w+)', context)
if match:
    print(f"Found: {match.group(1)}")
"""
        result = execute_code(code, globals_dict)

        assert result.success is True
        assert "XYZ789" in result.stdout
