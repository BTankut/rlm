"""Parser for LLM output - code blocks and FINAL detection"""

import re
from typing import Optional


def extract_code_blocks(text: str) -> list[str]:
    """
    Extract code blocks from LLM response.

    Looks for ```repl or ```python code blocks.

    Args:
        text: LLM response text

    Returns:
        List of code strings (without the backticks)
    """
    # Pattern to match ```repl or ```python blocks
    # Using non-greedy match and DOTALL for multiline
    pattern = r'```(?:repl|python)\s*\n(.*?)```'
    matches = re.findall(pattern, text, re.DOTALL)
    return matches


def detect_final_answer(text: str) -> tuple[bool, Optional[str]]:
    """
    Detect FINAL(answer) pattern in text.

    Args:
        text: LLM response text

    Returns:
        Tuple of (found: bool, answer: str or None)

    Note:
        - Answer can contain nested parentheses
        - Answer can be multiline
        - Should not detect FINAL inside code blocks
    """
    # First, remove code blocks to avoid false positives
    text_without_code = re.sub(r'```.*?```', '', text, flags=re.DOTALL)

    # Pattern for FINAL(...) with nested parentheses support
    # We'll use a simple approach: find FINAL( and then balance parentheses
    final_match = re.search(r'FINAL\s*\(', text_without_code)
    if not final_match:
        return False, None

    # Find the matching closing parenthesis
    start_idx = final_match.end()
    paren_count = 1
    idx = start_idx

    while idx < len(text_without_code) and paren_count > 0:
        char = text_without_code[idx]
        if char == '(':
            paren_count += 1
        elif char == ')':
            paren_count -= 1
        idx += 1

    if paren_count == 0:
        # Extract the content between FINAL( and the matching )
        answer = text_without_code[start_idx:idx - 1].strip()
        return True, answer

    return False, None


def detect_final_var(text: str) -> tuple[bool, Optional[str]]:
    """
    Detect FINAL_VAR(variable_name) pattern in text.

    Args:
        text: LLM response text

    Returns:
        Tuple of (found: bool, variable_name: str or None)
    """
    # Remove code blocks to avoid false positives
    text_without_code = re.sub(r'```.*?```', '', text, flags=re.DOTALL)

    # Pattern for FINAL_VAR(variable_name)
    # Variable names are typically alphanumeric with underscores
    pattern = r'FINAL_VAR\s*\(\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\)'
    match = re.search(pattern, text_without_code)

    if match:
        return True, match.group(1)

    return False, None


def has_code_block(text: str) -> bool:
    """Check if text contains any code block"""
    return bool(re.search(r'```(?:repl|python)', text))


def has_final(text: str) -> bool:
    """Check if text contains FINAL() or FINAL_VAR()"""
    text_without_code = re.sub(r'```.*?```', '', text, flags=re.DOTALL)
    return bool(re.search(r'FINAL(?:_VAR)?\s*\(', text_without_code))
