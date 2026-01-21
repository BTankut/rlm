#!/usr/bin/env python3
"""
RLM CLI - Command Line Interface for Recursive Language Models
"""

import argparse
import sys
import json
import logging
from pathlib import Path

from rlm.rlm import RLM
from rlm.config import Config


def setup_logging(debug: bool = False):
    """Set up logging configuration"""
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )


def run_query(args):
    """Run a single query"""
    # Load context
    if args.context_file:
        context_path = Path(args.context_file)
        if not context_path.exists():
            print(f"Error: Context file not found: {args.context_file}")
            sys.exit(1)
        context = context_path.read_text()
    elif args.context:
        context = args.context
    else:
        print("Error: Either --context or --context-file is required")
        sys.exit(1)

    # Create config
    config = Config()
    if args.model:
        config.root_model = args.model
        config.sub_model = args.model
    if args.max_iterations:
        config.max_iterations = args.max_iterations
    if args.no_subcalls:
        config.enable_sub_calls = False
    config.debug = args.debug

    # Run RLM
    rlm = RLM(config)
    print(f"Running query with model: {config.root_model}")
    print(f"Context size: {len(context):,} characters")
    print("-" * 50)

    result = rlm.run(query=args.query, context=context)

    # Output result
    print("-" * 50)
    print(f"Success: {result.success}")
    print(f"Iterations: {result.iterations}")
    print(f"Sub-calls: {result.sub_calls}")
    print(f"\nAnswer:\n{result.answer}")

    if result.error:
        print(f"\nError: {result.error}")

    if args.stats:
        print(f"\nUsage Statistics:")
        for key, value in result.usage_stats.items():
            print(f"  {key}: {value}")

    if args.json:
        output = {
            "success": result.success,
            "answer": result.answer,
            "iterations": result.iterations,
            "sub_calls": result.sub_calls,
            "error": result.error,
            "usage_stats": result.usage_stats
        }
        print(f"\nJSON Output:\n{json.dumps(output, indent=2)}")

    return 0 if result.success else 1


def run_interactive(args):
    """Run interactive mode"""
    config = Config()
    if args.model:
        config.root_model = args.model
        config.sub_model = args.model
    config.debug = args.debug

    rlm = RLM(config)

    print("RLM Interactive Mode")
    print(f"Model: {config.root_model}")
    print("Type 'quit' to exit, 'context <file>' to load context")
    print("-" * 50)

    context = ""

    while True:
        try:
            user_input = input("\nrlm> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() == 'quit':
            print("Goodbye!")
            break

        if user_input.lower().startswith('context '):
            filepath = user_input[8:].strip()
            try:
                context = Path(filepath).read_text()
                print(f"Loaded context: {len(context):,} characters")
            except Exception as e:
                print(f"Error loading context: {e}")
            continue

        if not context:
            print("No context loaded. Use 'context <file>' to load one.")
            continue

        print("Processing...")
        result = rlm.run(query=user_input, context=context)

        print(f"\n[Iterations: {result.iterations}, Sub-calls: {result.sub_calls}]")
        print(f"Answer: {result.answer}")

        if result.error:
            print(f"Error: {result.error}")


def run_benchmark(args):
    """Run benchmark suite"""
    config = Config()
    if args.model:
        config.root_model = args.model
        config.sub_model = args.model

    benchmark_name = args.name.lower()

    if benchmark_name == 's_niah':
        from benchmarks.s_niah import run_s_niah_benchmark

        print("Running S-NIAH Benchmark")
        print("-" * 50)

        sizes = [int(s.replace('k', '000').replace('m', '000000')) for s in args.sizes.split(',')]
        results = run_s_niah_benchmark(sizes=sizes, config=config)

        print("\nResults:")
        for size, accuracy in results.items():
            print(f"  {size:,} chars: {accuracy:.1%} accuracy")

    elif benchmark_name == 'oolong':
        from benchmarks.oolong import run_oolong_benchmark, print_oolong_report

        print("Running OOLONG Benchmark")
        print("-" * 50)

        sizes = [int(s.replace('k', '000').replace('m', '000000')) for s in args.sizes.split(',')]
        results = run_oolong_benchmark(config=config, context_sizes=sizes)
        print_oolong_report(results)

    elif benchmark_name == 'code_qa':
        from benchmarks.code_qa import run_code_qa_benchmark, print_code_qa_report

        print("Running Code QA Benchmark")
        print("-" * 50)

        results = run_code_qa_benchmark(config=config)
        print_code_qa_report(results)

    else:
        print(f"Unknown benchmark: {args.name}")
        print("Available benchmarks: s_niah, oolong, code_qa")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="RLM - Recursive Language Models CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run a query with context file
  python cli.py query --query "What is the answer?" --context-file document.txt

  # Run with specific model
  python cli.py query --query "Summarize this" --context-file doc.txt --model qwen3:8b

  # Interactive mode
  python cli.py interactive

  # Run benchmarks
  python cli.py benchmark --sizes 8k,16k,32k
        """
    )

    parser.add_argument('--debug', action='store_true', help='Enable debug output')

    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # Query command
    query_parser = subparsers.add_parser('query', help='Run a single query')
    query_parser.add_argument('--query', '-q', required=True, help='The query to run')
    query_parser.add_argument('--context', '-c', help='Context string')
    query_parser.add_argument('--context-file', '-f', help='Path to context file')
    query_parser.add_argument('--model', '-m', help='Model to use')
    query_parser.add_argument('--max-iterations', type=int, help='Maximum iterations')
    query_parser.add_argument('--no-subcalls', action='store_true', help='Disable sub-calls')
    query_parser.add_argument('--stats', action='store_true', help='Show usage statistics')
    query_parser.add_argument('--json', action='store_true', help='Output as JSON')

    # Interactive command
    interactive_parser = subparsers.add_parser('interactive', help='Interactive mode')
    interactive_parser.add_argument('--model', '-m', help='Model to use')

    # Benchmark command
    benchmark_parser = subparsers.add_parser('benchmark', help='Run benchmarks')
    benchmark_parser.add_argument('--name', '-n', default='s_niah',
                                  choices=['s_niah', 'oolong', 'code_qa'],
                                  help='Benchmark name (s_niah, oolong, code_qa)')
    benchmark_parser.add_argument('--sizes', default='8k,16k,32k',
                                  help='Context sizes (e.g., 8k,16k,32k or 100k,1m)')
    benchmark_parser.add_argument('--model', '-m', help='Model to use')

    args = parser.parse_args()

    setup_logging(args.debug)

    if args.command == 'query':
        sys.exit(run_query(args))
    elif args.command == 'interactive':
        run_interactive(args)
    elif args.command == 'benchmark':
        run_benchmark(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
