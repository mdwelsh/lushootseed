#!/usr/bin/env python3

import argparse
import csv
import random
from typing import List, Tuple

from openai import OpenAI

openai_client = OpenAI()


def query_openai(prompt: str) -> str:
    completion = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
    )
    return completion.choices[0].message


def eval_one(prompt: str, lut: str, eng: str) -> float:
    full_prompt = prompt + lut + "\n"
    answer = query_openai(full_prompt)
    print("\n" + lut)
    print(f"Expected: {eng}")
    print(f"Got: {answer}")
    return 0.0


def do_eval(
    prompt_examples: List[Tuple[str, str]], test_examples: List[Tuple[str, str]]
):
    prompt = """Your task is to translate a phrase provided by the user in the
    Lushootseed language to English. To help with this task, you are provided with
    the following list of phrases in both Luhshootseed and English. You can use
    this list to help you translate the user's phrase. Please ONLY provide the English
    translation of the user's phrase. If you are unsure, try to use the information provided
    to make an educated guess.\n
    """

    for lut, eng in prompt_examples:
        prompt += f"Lushootseed: {lut}\n"
        prompt += f"English: {eng}\n\n"

    prompt += "\nPlease translate the following Lushootseed phrase to English:\n"
    for lut, eng in test_examples:
        eval_one(prompt, lut, eng)
        break


def main():
    parser = argparse.ArgumentParser(
        prog="eval",
        description="Evaluate Lushootseed-English translation",
    )
    parser.add_argument(
        "filename",
        type=str,
        help="CSV file containing Lushootseed-English translations",
    )
    parser.add_argument(
        "-f", type=float, default=0.05, help="Fraction of entries to test"
    )
    args = parser.parse_args()

    with open("raw.csv", "r") as f:
        reader = csv.DictReader(f)
        examples = []
        for _, row in enumerate(reader):
            lut = row["Lushootseed"]
            eng = row["English"]
            examples.append((lut, eng))
        # Limit number of examples.
        examples = examples[:1000]
        # Shuffle examples.
        random.shuffle(examples)
        n = len(examples)
        n_test = int(args.f * n)
        test_examples = examples[:n_test]
        prompt_examples = examples[n_test:]
        do_eval(prompt_examples, test_examples)


# print(query_openai("How many dogs can fit into a Volkswagen bus?"))


if __name__ == "__main__":
    main()
