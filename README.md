# RLM - Recursive Language Models

**Paper-aligned minimal implementation** of the RLM approach from [Zhang et al. (2025)](https://arxiv.org/abs/2512.24601).

Process long contexts (100K+ chars, tested up to 1M+) via iterative REPL execution.

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

# Or chunk and analyze large contexts with sub-LLM calls
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
python cli.py benchmark --name oolong_pairs --tasks 1,2,3
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
| EXECUTION_TIMEOUT | 30 | Code timeout (seconds) |
| ENABLE_SUB_CALLS | true | Enable llm_query() for sub-calls |
| MAX_SUB_CALLS | 100 | Sub-call limit |
| MAX_SUB_CALL_CHARS | 500000 | Max chars per sub-call |
| TEMPERATURE | 0.7 | Sampling temperature |
| DEBUG | false | Verbose logging |

## Project Structure

```
.
├── cli.py               # CLI entry point
├── main.py              # Demo runner
├── benchmarks/          # Benchmarks
│   ├── s_niah.py        # Single Needle in a Haystack
│   ├── oolong.py        # OOLONG benchmark
│   ├── oolong_pairs.py  # OOLONG-Pairs (paper-aligned, 20 tasks)
│   └── code_qa.py       # Code Q&A benchmark
├── tests/               # Unit + integration tests
└── rlm/
    ├── config.py        # Environment configuration
    ├── rlm.py           # Main orchestrator (paper-aligned)
    ├── core/
    │   ├── executor.py  # Code execution with timeout
    │   ├── parser.py    # Extract code blocks and FINAL()
    │   ├── state.py     # REPL state management
    │   └── tracking.py  # Usage statistics
    └── llm/
        ├── client.py    # Ollama client
        └── prompts.py   # System prompts (paper-aligned, Appendix D)
```

## Paper-Aligned Implementation

This implementation follows the paper exactly:

1. **System Prompts**: Copied verbatim from Appendix D (sec3-methods.tex)
   - RLM with REPL (sub-calls): Lines 5-77
   - RLM with REPL (no sub-calls): Lines 89-142
   - Qwen diff for sub-call warning: Lines 79-86

2. **Placeholders**: `{context_type}`, `{context_total_length}`, `{context_lengths}`

3. **OOLONG-Pairs Benchmark**: All 20 tasks from Appendix E.1 (sec4-benchmarks.tex)

## Testing

```bash
# Fast tests only (no LLM calls)
pytest -m "not slow" tests/

# Slow tests (requires Ollama)
pytest -m "slow" tests/

# Specific test
pytest tests/test_rlm.py::TestRLMNeedleInHaystack::test_needle_100k -v
pytest tests/test_1m_context.py::TestMillionCharContext::test_1m_context_handling -v
pytest tests/test_oolong_pairs.py -v

# With coverage
pytest --cov=rlm tests/
```

## Requirements

- Python 3.10+
- Ollama running locally
- 8GB+ VRAM recommended (for qwen3:8b)

## References

- [Recursive Language Models](https://arxiv.org/abs/2512.24601) - Zhang et al. (2025)
- Paper PDF included: `paper.pdf`

## License

MIT
