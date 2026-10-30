
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
You are extracting evidence for an organizational science
study of historical urban planning decisions.

Your goal is NOT to summarize the document.
Only extract statements that help reconstruct:
- what decision-makers believed,
- what choices they considered,
- how they evaluated those choices,
- and why they selected or rejected them.

Use exactly these categories:

assumption:
A belief about conditions that supports a planning choice.

alternative:
A specific, identifiable course of action considered
by decision-makers. General descriptions of the study
area do NOT qualify.

constraint:
An explicit limitation on what decision-makers could do,
such as cost, legal restrictions, engineering limitations,
or right-of-way availability.

forecast:
A prediction about future conditions or outcomes.
Descriptions of planned procedures do NOT qualify.

evaluation_criterion:
A factor explicitly used to compare choices,
such as cost, ridership, environmental impacts,
or travel time.

decision_rationale:
An explicitly documented reason for selecting,
preferring, modifying, or rejecting an alternative.

IMPORTANT EXCLUSIONS:
- Do not extract generic background descriptions.
- Do not extract descriptions of the planning process.
- Do not confuse design characteristics with constraints.
- Do not interpret statements of intention as forecasts.
- Do not infer undocumented motivations.
- Do not manufacture claims to fill categories.

For each finding:
- Identify one specific, substantive claim.
- Copy the exact supporting quotation from the input.
- Use only the supplied document text.
- If no qualifying findings exist, return an empty list.
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


def main(doc_path="./documents/d2deismarch2010.pdf", start_page=1, pages=5):

    path = Path(doc_path)
    output = Path("outputs/findings.json")
    output.parent.mkdir(parents=True, exist_ok=True)

    results = []

    with fitz.open(path) as pdf:
        start = start_page - 1
        end = min(start + pages, len(pdf))

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
