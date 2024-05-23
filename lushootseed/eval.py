#!/usr/bin/env python3

import argparse
import csv
import random
from typing import List, Tuple

import nltk.translate.bleu_score as bleu
from openai import OpenAI
from rich.console import Console


console = Console()
openai_client = OpenAI()

total_input_tokens = 0
total_output_tokens = 0

# This is for GPT-4o.
PRICE_PER_INPUT_TOKEN = 5.0 / 1e6
PRICE_PER_OUTPUT_TOKEN = 15.0 / 1e6


def query_openai(prompt: str) -> str:
    global total_input_tokens
    global total_output_tokens

    completion = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
    )
    total_input_tokens += completion.usage.prompt_tokens
    total_output_tokens += completion.usage.completion_tokens
    return str(completion.choices[0].message.content)


def translate_one(prompt: str, lut: str) -> str:
    full_prompt = prompt + lut + "\n"
    answer = query_openai(full_prompt)
    return answer


def do_eval(
    prompt_examples: List[Tuple[str, str]], test_examples: List[Tuple[str, str]]
):
    expected_translations = []
    got_translations = []
    bleu_scores = []
    global total_input_tokens
    global total_output_tokens

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
    for i, (lut, eng) in enumerate(test_examples):
        console.rule(f"{i}/{len(test_examples)}")
        console.print(f"[{i}/{len(test_examples)}] Lushootseed: [green]{lut}[/]")
        console.print(f"[{i}/{len(test_examples)}] Expected:    [gray]{eng}[/]")
        got = translate_one(prompt, lut)
        console.print(f"[{i}/{len(test_examples)}] Got:         [yellow]{got}[/]")
        expected_translations.append([eng.strip().split()])
        got_translations.append(got.strip().split())
        bleu_score = bleu.sentence_bleu([eng.strip().split()], got.strip().split())
        console.print(f"[{i}/{len(test_examples)}] BLEU:        [blue]{bleu_score}[/]")
        bleu_scores.append(bleu_score)

    overall_bleu = bleu.corpus_bleu(expected_translations, got_translations)
    console.print(f"Overall BLEU score: [bold blue]{overall_bleu}[/]")
    console.print(f"Total prompt token usage: [red]{total_input_tokens}[/]")
    console.print(f"Total completion token usage: [red]{total_output_tokens}[/]")
    console.print(
        f"This little experiment cost you [red]${total_input_tokens * PRICE_PER_INPUT_TOKEN + total_output_tokens * PRICE_PER_OUTPUT_TOKEN}[/]."
    )


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
        "-f", type=float, default=0.01, help="Fraction of entries to test"
    )
    args = parser.parse_args()

    all_examples = {}
    with open(args.filename, "r") as f:
        reader = csv.DictReader(f)
        for _, row in enumerate(reader):
            lut = row["Lushootseed"]
            eng = row["English"]
            # To avoid duplication, we only retain a single English translation for
            # each Lushootseed phrase.
            all_examples[lut] = eng

    console.print(f"Loaded {len(all_examples)} examples")
    examples = [(lut, eng) for lut, eng in all_examples.items()]
    # Limit number of examples.
    examples = examples[:2500]
    # Shuffle examples.
    random.shuffle(examples)
    n = len(examples)
    n_test = int(args.f * n)
    test_examples = examples[0:n_test]
    prompt_examples = examples[n_test:]
    do_eval(prompt_examples, test_examples)


# print(query_openai("How many dogs can fit into a Volkswagen bus?"))


if __name__ == "__main__":
    main()
