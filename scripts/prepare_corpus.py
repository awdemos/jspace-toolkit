"""Generate a small pretraining-style prompt corpus."""

import argparse
import json
import random
import sys

from jspace import JSpaceError
from jspace.validation import validate_path, validate_workspace

PROMPT_TEMPLATES = [
    "The {noun} {verb} over the {adj} {noun2}.",
    "In 19{year}, scientists discovered that {noun} can {verb}.",
    "{question} The answer is",
    "Once upon a time, a {adj} {noun} decided to",
]

NOUNS = ["cat", "robot", "river", "piano", "astronaut", "theorem", "forest"]
VERBS = ["jumped", "sang", "calculated", "whispered", "exploded", "meandered"]
ADJS = ["curious", "ancient", "silent", "brilliant", "fierce", "gigantic"]
QUESTIONS = [
    "What color is the sky?",
    "Who wrote Hamlet?",
    "How many legs does a spider have?",
    "What is the capital of France?",
]


def generate(n: int) -> list:
    prompts = []
    for _ in range(n):
        # These random choices are for reproducible demo data, not security.  # nosec B311
        template = random.choice(PROMPT_TEMPLATES)  # nosec B311
        prompts.append(
            template.format(
                noun=random.choice(NOUNS),  # nosec B311
                noun2=random.choice(NOUNS),  # nosec B311
                verb=random.choice(VERBS),  # nosec B311
                adj=random.choice(ADJS),  # nosec B311
                year=random.randint(50, 99),  # nosec B311
                question=random.choice(QUESTIONS),  # nosec B311
            )
        )
    return prompts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=1024)
    parser.add_argument("--out", default="corpus.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--workspace",
        default=".",
        help="Root directory that --out must be contained within",
    )
    args = parser.parse_args()
    random.seed(args.seed)
    try:
        workspace = validate_workspace(args.workspace)
        out_path = validate_path(args.out, workspace)
        with out_path.open("w") as f:
            json.dump(generate(args.n), f)
    except (OSError, JSpaceError) as exc:
        print(f"Error writing corpus to {args.out}: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
