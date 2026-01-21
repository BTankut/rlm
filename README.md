# RLM - Recursive Language Models

Process long contexts (100K+ chars, tested up to 1M+ chars; LLM dependent) via iterative REPL execution.

Based on the approach from [Zhang et al. (2025)](https://arxiv.org/abs/2512.24601) - instead of passing entire contexts to an LLM, RLM provides a REPL environment where the LLM can iteratively execute Python code to explore, analyze, and query the context programmatically.

## How It Works

```
User Query + Context
        │
        ▼
   Context stored in REPL as 'context' variable
        │
        ▼
┌─────────────────────────────────┐
│       Iteration Loop            │
│  1. LLM generates code          │
│  2. Code executes in REPL       │
│  3. Output sent back to LLM     │
│  4. Repeat until FINAL()        │
└─────────────────────────────────┘
        │
        ▼
      Answer
```

The LLM writes Python code to search, analyze, and process the context:

```python
# LLM can use regex to find patterns
import re
m = re.search(r'\[SECRET: (\w+)\]', context)
print(m.group(1))

# Or chunk and analyze large contexts
chunks = [context[i:i+100000] for i in range(0, len(context), 100000)]
results = [llm_query(f"Summarize: {chunk}") for chunk in chunks]
```

## Installation

```bash
# Clone repository
git clone https://github.com/BTankut/rlm.git
cd rlm

# Install dependencies
pip install -r requirements.txt

# Start Ollama (required)
ollama serve

# Pull a model
ollama pull qwen3:8b  # or qwen3:14b for better results
```

## Quick Start

```bash
# Simple query with context file
python cli.py query -q "What is the main topic?" -f document.txt

# Interactive mode
python cli.py interactive

# Run demo
python main.py --test simple
python main.py --test needle --size 50000
```

## CLI Usage

```bash
# Query with specific model
python cli.py query -q "Summarize this" -f doc.txt --model qwen3:8b

# Run benchmarks
python cli.py benchmark --name s_niah --sizes 8k,16k,32k
python cli.py benchmark --name oolong --sizes 50k,100k
python cli.py benchmark --name code_qa
```

## Configuration

Create a `.env` file (see `.env.example`):

| Variable | Default | Description |
|----------|---------|-------------|
| OLLAMA_BASE_URL | http://localhost:11434/v1 | Ollama API endpoint |
| ROOT_MODEL | qwen3:14b | Main LLM model |
| SUB_MODEL | (same as ROOT_MODEL) | Model for sub-calls |
| MAX_ITERATIONS | 20 | Loop safety limit |
| MAX_OUTPUT_CHARS | 10000 | Truncation limit |
| MAX_CONTEXT_CHARS_DISPLAY | 500 | Context preview size |
| EXECUTION_TIMEOUT | 30 | Code timeout (seconds) |
| ENABLE_SUB_CALLS | true | Enable llm_query() for sub-calls |
| MAX_SUB_CALLS | 100 | Sub-call limit |
| SUB_CALL_WARNING_THRESHOLD | 50 | Warning threshold for sub-calls |
| MAX_SUB_CALL_CHARS | 500000 | Max chars per sub-call |
| TEMPERATURE | 0.7 | Sampling temperature |
| DEBUG | false | Verbose logging |
| USE_SANDBOX | false | Use RestrictedPython sandbox |
| SANDBOX_USE_RESTRICTED_PYTHON | true | Use RestrictedPython if available |
| MAX_LLM_RETRIES | 3 | LLM retry attempts |
| RETRY_BACKOFF_BASE | 2.0 | Backoff base for retries |
| MAX_EXECUTION_RETRIES | 3 | Max execution retry attempts |

## Project Structure

```
.
├── cli.py               # CLI entry point
├── main.py              # Demo runner
├── benchmarks/          # Benchmarks (s_niah, oolong, code_qa)
├── tests/               # Unit + slow integration tests
└── rlm/
    ├── config.py        # Environment configuration
    ├── rlm.py           # Main orchestrator
    ├── core/
    │   ├── executor.py  # Code execution with timeout
    │   ├── sandbox.py   # Optional sandboxing
    │   ├── parser.py    # Extract code blocks and FINAL()
    │   ├── state.py     # REPL state management
    │   ├── tracking.py  # Usage statistics
    │   ├── cache.py     # LLM response cache
    │   └── logging.py   # Structured logging
    └── llm/
        ├── client.py        # Ollama client
        ├── async_client.py  # Async client for parallel calls
        └── prompts.py       # System prompts
```

## Testing

```bash
# Fast tests only (no LLM calls)
pytest -m "not slow" tests/

# Slow tests (requires Ollama, can take minutes)
pytest -m "slow" tests/

# Specific test
pytest tests/test_rlm.py::TestRLMNeedleInHaystack::test_needle_100k -v
pytest tests/test_1m_context.py::TestMillionCharContext::test_1m_context_handling -v

# With coverage
pytest --cov=rlm tests/
```

## Requirements

- Python 3.10+
- Ollama running locally
- 8GB+ VRAM recommended (for qwen3:8b)

## References

- [Recursive Language Models](https://arxiv.org/abs/2512.24601) - Zhang et al. (2025)

## License

MIT
