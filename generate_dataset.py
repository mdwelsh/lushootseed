#!/usr/bin/env python3

import csv
import json

LUT2ENG_PROMPT = """Your task is to help the user translate Lushootseed phrases to English.
The user will provide you with a Lushootseed phrase, and you will provide the English translation.
If you do not know the translation, you can say 'I don't know'."""

ENG2LUT_PROMPT = """Your task is to help the user translate English phrases to Lushootseed.
The user will provide you with an English phrase, and you will provide the Lushootseed translation.
If you do not know the translation, you can say 'I don't know'."""


def gen_prompt(sysprompt: str, fromstr: str, tostr: str) -> str:
    p = {
        "messages": [
            {"role": "system", "content": sysprompt},
            {"role": "user", "content": fromstr},
            {"role": "assistant", "content": tostr},
        ]
    }
    return json.dumps(p)


with open("train.jsonl", "w") as trainout:
    with open("test.jsonl", "w") as testout:
        with open("raw.csv", "r") as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                lut = row["Lushootseed"]
                eng = row["English"]
                prompt = gen_prompt(LUT2ENG_PROMPT, lut, eng)
                if i % 5 == 0:
                    testout.write(prompt + "\n")
                else:
                    trainout.write(prompt + "\n")
                prompt = gen_prompt(ENG2LUT_PROMPT, eng, lut)
                if i % 5 == 3:
                    testout.write(prompt + "\n")
                else:
                    trainout.write(prompt + "\n")
