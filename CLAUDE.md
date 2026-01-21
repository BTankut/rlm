# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

RLM (Recursive Language Models) implements the approach from Zhang et al. (2025) for processing extremely long contexts (10M+ tokens). Instead of passing entire contexts to an LLM, RLM provides a REPL environment where the LLM can iteratively execute Python code to explore, analyze, and query the context programmatically.

## Build and Test Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run all tests (fast unit tests only)
pytest -m "not slow" tests/

# Run all tests including slow LLM-dependent tests
pytest tests/

# Run specific test file
pytest tests/test_parser.py

# Run single test
pytest tests/test_rlm.py::TestRLMIntegration::test_simple_query -v

# Run with coverage
pytest --cov=rlm tests/

# Run demo
python main.py --test simple
python main.py --test needle --size 50000
```

## CLI Usage

```bash
# Query with context file
python cli.py query -q "Find the answer" -f document.txt

# With specific model
python cli.py query -q "Summarize" -f doc.txt --model qwen3:8b

# Interactive mode
python cli.py interactive

# Benchmarks
python cli.py benchmark --name s_niah --sizes 8k,16k,32k
python cli.py benchmark --name oolong --sizes 50k,100k
```

## Architecture

### Core Loop (rlm/rlm.py)

The `RLM` class orchestrates the iterative LLM-REPL interaction:

1. Initialize `REPLState` with context as a `context` variable
2. Build system prompt with context metadata
3. Call LLM with conversation history
4. Parse response for code blocks (` ```repl `) or `FINAL(answer)`
5. Execute code blocks, capture output
6. Truncate output (40% head + 40% tail) and append to conversation
7. Repeat until `FINAL()` detected or max iterations reached

### Key Components

```
rlm/
├── rlm.py              # Main orchestrator - RLM class and run() method
├── config.py           # Environment-based configuration via .env
├── errors.py           # Custom exceptions (RLMError, MaxIterationsError, etc.)
├── core/
│   ├── executor.py     # exec() with timeout and stdout capture
│   ├── parser.py       # Regex extraction of code blocks and FINAL()
│   ├── state.py        # REPLState - globals dict + execution history
│   ├── sandbox.py      # RestrictedPython sandboxing (optional)
│   ├── tracking.py     # UsageTracker - token/call statistics
│   ├── cache.py        # LRU cache for LLM responses
│   └── logging.py      # Structured JSON logging
└── llm/
    ├── client.py       # OpenAI SDK client for Ollama
    ├── async_client.py # Async client for parallel sub-calls
    └── prompts.py      # System prompts (Phase 1 basic, Phase 2 with llm_query)
```

### Data Flow

```
User Query + Context
        │
        ▼
    REPLState.initialize_context(context)  # context available as variable
        │
        ▼
    System Prompt (from prompts.py)
        │
        ▼
┌───────────────────────────────────────┐
│           Iteration Loop              │
│  ┌─────────────────────────────────┐  │
│  │ LLM generates code or FINAL()  │  │
│  └──────────────┬──────────────────┘  │
│                 │                     │
│       ┌─────────▼─────────┐           │
│       │ parser.py extracts│           │
│       │ code blocks       │           │
│       └─────────┬─────────┘           │
│                 │                     │
│       ┌─────────▼─────────┐           │
│       │ executor.py runs  │           │
│       │ code in REPLState │           │
│       └─────────┬─────────┘           │
│                 │                     │
│       ┌─────────▼─────────┐           │
│       │ Output truncated, │           │
│       │ added to messages │           │
│       └─────────┬─────────┘           │
│                 │                     │
└─────────────────┴─────────────────────┘
        │
        ▼
    RLMResult(answer, iterations, success, execution_history)
```

### Phase 2: Sub-Calls

When `enable_sub_calls=True`, the REPL environment includes `llm_query(prompt) -> str` function. This allows the LLM to:
- Chunk large contexts
- Query sub-LLM on each chunk
- Aggregate results

Pattern from paper:
```python
chunks = [context[i:i+100000] for i in range(0, len(context), 100000)]
results = [llm_query(f"Analyze: {chunk}") for chunk in chunks]
final = llm_query(f"Combine: {results}")
```

## Configuration

Environment variables in `.env` (see `config.py`):

| Variable | Default | Description |
|----------|---------|-------------|
| OLLAMA_BASE_URL | http://localhost:11434/v1 | Ollama API endpoint |
| ROOT_MODEL | qwen3:14b | Main LLM model |
| SUB_MODEL | (same as ROOT_MODEL) | Model for sub-calls |
| MAX_ITERATIONS | 20 | Loop safety limit |
| MAX_OUTPUT_CHARS | 10000 | Truncation limit |
| EXECUTION_TIMEOUT | 30 | Code timeout (seconds) |
| MAX_SUB_CALLS | 100 | Sub-call limit |
| ENABLE_SUB_CALLS | true | Enable llm_query() |
| USE_SANDBOX | false | Enable RestrictedPython |
| DEBUG | false | Verbose logging |

## Test Markers

Tests use pytest markers:
- `@pytest.mark.slow` - LLM-dependent integration tests (skip with `-m "not slow"`)

## Prerequisites

- Ollama running: `ollama serve`
- Model pulled: `ollama pull qwen3:14b` (or qwen3:8b for 8GB VRAM)
- Python 3.10+
