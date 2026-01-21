"""Tests for the parser module"""

import pytest
from rlm.core.parser import (
    extract_code_blocks,
    detect_final_answer,
    detect_final_var,
    has_code_block,
    has_final
)


class TestExtractCodeBlocks:
    """Tests for extract_code_blocks function"""

    def test_single_repl_block(self):
        text = """Some text
```repl
print("hello")
```
More text"""
        blocks = extract_code_blocks(text)
        assert len(blocks) == 1
        assert 'print("hello")' in blocks[0]

    def test_single_python_block(self):
        text = """Some text
```python
x = 42
print(x)
```
More text"""
        blocks = extract_code_blocks(text)
        assert len(blocks) == 1
        assert 'x = 42' in blocks[0]

    def test_multiple_blocks(self):
        text = """
```repl
x = 1
```

```python
y = 2
```

```repl
z = 3
```
"""
        blocks = extract_code_blocks(text)
        assert len(blocks) == 3

    def test_no_blocks(self):
        text = "Just plain text without any code blocks"
        blocks = extract_code_blocks(text)
        assert len(blocks) == 0

    def test_ignores_other_languages(self):
        text = """
```javascript
console.log("hello");
```

```repl
print("hello")
```
"""
        blocks = extract_code_blocks(text)
        assert len(blocks) == 1
        assert 'print("hello")' in blocks[0]

    def test_multiline_code(self):
        text = """
```repl
for i in range(10):
    print(i)
    if i > 5:
        break
```
"""
        blocks = extract_code_blocks(text)
        assert len(blocks) == 1
        assert 'for i in range(10):' in blocks[0]
        assert 'if i > 5:' in blocks[0]


class TestDetectFinalAnswer:
    """Tests for detect_final_answer function"""

    def test_simple_final(self):
        text = "The answer is FINAL(42)"
        found, answer = detect_final_answer(text)
        assert found is True
        assert answer == "42"

    def test_final_with_text(self):
        text = "Based on my analysis, FINAL(The secret code is ABC123)"
        found, answer = detect_final_answer(text)
        assert found is True
        assert answer == "The secret code is ABC123"

    def test_final_with_parentheses(self):
        text = "FINAL(The function f(x) returns x + 1)"
        found, answer = detect_final_answer(text)
        assert found is True
        assert "f(x)" in answer

    def test_final_multiline(self):
        text = """FINAL(Line 1
Line 2
Line 3)"""
        found, answer = detect_final_answer(text)
        assert found is True
        assert "Line 1" in answer
        assert "Line 3" in answer

    def test_no_final(self):
        text = "This text has no final answer"
        found, answer = detect_final_answer(text)
        assert found is False
        assert answer is None

    def test_final_inside_code_block_ignored(self):
        text = """
```repl
result = "FINAL(not this one)"
print(result)
```
FINAL(This is the real answer)
"""
        found, answer = detect_final_answer(text)
        assert found is True
        assert answer == "This is the real answer"

    def test_final_with_space(self):
        text = "FINAL (the answer)"
        found, answer = detect_final_answer(text)
        assert found is True
        assert answer == "the answer"


class TestDetectFinalVar:
    """Tests for detect_final_var function"""

    def test_simple_final_var(self):
        text = "FINAL_VAR(result)"
        found, var_name = detect_final_var(text)
        assert found is True
        assert var_name == "result"

    def test_final_var_with_underscore(self):
        text = "FINAL_VAR(my_result_variable)"
        found, var_name = detect_final_var(text)
        assert found is True
        assert var_name == "my_result_variable"

    def test_final_var_with_numbers(self):
        text = "FINAL_VAR(result123)"
        found, var_name = detect_final_var(text)
        assert found is True
        assert var_name == "result123"

    def test_no_final_var(self):
        text = "No FINAL_VAR here"
        found, var_name = detect_final_var(text)
        assert found is False
        assert var_name is None

    def test_final_var_inside_code_block_ignored(self):
        text = """
```repl
# FINAL_VAR(not_this)
```
FINAL_VAR(real_var)
"""
        found, var_name = detect_final_var(text)
        assert found is True
        assert var_name == "real_var"


class TestHasCodeBlock:
    """Tests for has_code_block function"""

    def test_has_repl_block(self):
        text = "```repl\ncode\n```"
        assert has_code_block(text) is True

    def test_has_python_block(self):
        text = "```python\ncode\n```"
        assert has_code_block(text) is True

    def test_no_block(self):
        text = "plain text"
        assert has_code_block(text) is False

    def test_other_language_block(self):
        text = "```javascript\ncode\n```"
        assert has_code_block(text) is False


class TestHasFinal:
    """Tests for has_final function"""

    def test_has_final(self):
        text = "FINAL(answer)"
        assert has_final(text) is True

    def test_has_final_var(self):
        text = "FINAL_VAR(var)"
        assert has_final(text) is True

    def test_no_final(self):
        text = "no final here"
        assert has_final(text) is False

    def test_final_in_code_block_not_counted(self):
        text = "```repl\nFINAL(x)\n```"
        assert has_final(text) is False
