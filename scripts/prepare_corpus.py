"""Generate a small pretraining-style prompt corpus."""

import argparse
import json
import random
import sys
from pathlib import Path

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
        template = random.choice(PROMPT_TEMPLATES)
        prompts.append(
            template.format(
                noun=random.choice(NOUNS),
                noun2=random.choice(NOUNS),
                verb=random.choice(VERBS),
                adj=random.choice(ADJS),
                year=random.randint(50, 99),
                question=random.choice(QUESTIONS),
            )
        )
    return prompts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=1024)
    parser.add_argument("--out", default="corpus.json")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    random.seed(args.seed)
    try:
        out_path = Path(args.out)
        with out_path.open("w") as f:
            json.dump(generate(args.n), f)
    except OSError as exc:
        print(f"Error writing corpus to {args.out}: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
