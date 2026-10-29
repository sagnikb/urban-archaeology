
import argparse
import json
import re
from pathlib import Path

import pymupdf as fitz
from ollama import chat
from pydantic import BaseModel, Field


MODEL = "qwen3:4b"


class Finding(BaseModel):
    category: str = Field(
        description="One of: assumption, alternative, constraint, forecast"
    )
    claim: str
    evidence_quote: str


class Extraction(BaseModel):
    findings: list[Finding]


SYSTEM_PROMPT = """
You are a researcher studying how urban planning organizations
make infrastructure decisions.

Extract claims in these categories:

1. assumption: beliefs used to justify a planning decision
2. alternative: possible courses of action considered
3. constraint: limitations affecting the choice
4. forecast: predictions about future conditions or outcomes

Rules:
- Use only information explicitly stated in the supplied text.
- Do not infer intentions or hidden motivations.
- Copy an exact supporting quotation for each finding.
- Prefer concrete, research-useful claims over generic statements.
- Do not invent numerical values or historical facts.
- Return an empty list if the passage has no relevant evidence.
"""


def normalize(text):
    return re.sub(r"\s+", " ", text).strip()


def analyze(text, filename, page):
    response = chat(
        model=MODEL,
        think=False,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ],
        format=Extraction.model_json_schema(),
        options={
            "temperature": 0,
            "num_ctx": 8192,
        },
    )

    extraction = Extraction.model_validate_json(
        response.message.content
    )

    records = []

    for finding in extraction.findings:
        record = finding.model_dump()
        quote = normalize(finding.evidence_quote)

        record["document"] = filename
        record["page"] = page
        record["quote_verified"] = (
            bool(quote) and quote in normalize(text)
        )

        records.append(record)

    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("pdf")
    parser.add_argument("--start-page", type=int, default=1)
    parser.add_argument("--pages", type=int, default=5)
    args = parser.parse_args()

    path = Path(args.pdf)
    output = Path("outputs/findings.json")
    output.parent.mkdir(parents=True, exist_ok=True)

    results = []

    with fitz.open(path) as pdf:
        start = args.start_page - 1
        end = min(start + args.pages, len(pdf))

        if start < 0 or start >= len(pdf):
            raise ValueError("Start page is outside the PDF")

        for i in range(start, end):
            page = i + 1
            text = pdf[i].get_text(sort=True)

            print(f"Processing PDF page {page}...", flush=True)

            if len(text.strip()) < 100:
                print("  Skipped: insufficient text")
                continue

            # Keep prompts manageable for the local 4B model.
            # This first prototype only analyzes the first
            # 10,000 characters of each page.
            try:
                records = analyze(
                    text[:5000], path.name, page
                )
                results.extend(records)
                print(f"  Found {len(records)} claims")

            except Exception as error:
                print(f"  Error: {error}")

            # Save after every page to preserve progress.
            output.write_text(
                json.dumps(results, indent=2),
                encoding="utf-8",
            )

    print(f"\nTotal findings: {len(results)}")
    print(f"Saved to {output}")


if __name__ == "__main__":
    main()
