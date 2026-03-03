from __future__ import annotations

import argparse
import importlib
from collections import defaultdict

COMMAND_TARGETS: dict[tuple[str, str], tuple[str, str]] = {
    ("harm", "seed-preferences"): ("src.experiments.harm_kl.seed_preferences", "main"),
    ("harm", "generate-conversations"): ("src.experiments.harm_kl.generate_preferences", "main"),
    ("harm", "compute-drift"): ("src.experiments.harm_kl.compute_drift", "main"),
    ("harm", "plot"): ("src.experiments.harm_kl.plot", "main"),
    ("harm", "analyze-jumps"): ("src.experiments.harm_kl.analyze_jumps", "main"),
    ("harm", "plot-crossover"): ("src.experiments.harm_kl.plot_crossover", "main"),
    ("dpo", "train-hh-token"): ("src.experiments.dpo_hh_token.train", "main"),
    ("fairmt", "generate"): ("src.experiments.fairmt.generate", "main"),
    ("fairmt", "judge-heuristic"): ("src.experiments.fairmt.judge_compare", "main"),
    ("fairmt", "judge-pairwise-llm"): ("src.experiments.fairmt.judge_pairwise_llm", "main"),
    ("token", "expound-hh-token-delta"): ("src.experiments.token_semantics.expound_hh_token_delta", "main"),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Unified experiment runner. Usage: python -m src.run <suite> <command> [args...]"
    )
    suites: dict[str, list[str]] = defaultdict(list)
    for suite, command in COMMAND_TARGETS:
        suites[suite].append(command)

    suite_subparsers = parser.add_subparsers(dest="suite", required=True)
    for suite, commands in sorted(suites.items()):
        suite_parser = suite_subparsers.add_parser(suite)
        command_subparsers = suite_parser.add_subparsers(dest="command", required=True)
        for command in sorted(commands):
            command_parser = command_subparsers.add_parser(command)
            command_parser.add_argument("--dry-run", action="store_true")
            command_parser.add_argument("command_args", nargs=argparse.REMAINDER)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    key = (args.suite, args.command)
    if key not in COMMAND_TARGETS:
        parser.error(f"Unknown command: {args.suite} {args.command}")

    module_name, func_name = COMMAND_TARGETS[key]
    pass_through_args = list(args.command_args)

    if args.dry_run:
        print(f"[dry-run] would run: python -m src.run {args.suite} {args.command} {' '.join(pass_through_args)}")
        return 0

    module = importlib.import_module(module_name)
    func = getattr(module, func_name)
    func(pass_through_args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
