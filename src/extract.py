
import json
import re
from pathlib import Path

import pymupdf as fitz
from ollama import chat
from pydantic import BaseModel, Field

from typing import Literal


MODEL = "qwen3:4b"


class Finding(BaseModel):
    category: Literal[
        "assumption",
        "alternative",
        "constraint",
        "forecast",
        "evaluation_criterion",
        "decision_rationale",
        "project_objective",
    ]
    claim: str
    evidence_quote: str
    classification_reason: str


class Extraction(BaseModel):
    findings: list[Finding]




SYSTEM_PROMPT = """
You are a research assistant specializing in urban planning,
organizational decision-making, and historical document analysis.

Your task is to extract structured evidence from historical
planning documents for a research project called
"Urban Decision Archaeology."

RESEARCH OBJECTIVE
------------------
Reconstruct how public organizations made infrastructure
and urban planning decisions over time.

We are particularly interested in:
- What organizations believed when making decisions.
- Which alternatives were considered.
- What constraints affected their choices.
- What outcomes they anticipated.
- How they evaluated competing alternatives.
- Why they selected, rejected, or revised decisions.

Your task is NOT to summarize the document.

Extract only specific, substantive statements that provide
evidence about organizational decision-making.

CATEGORY DEFINITIONS
--------------------

1. assumption

A belief about existing or future conditions that an
organization explicitly uses to motivate, justify, or
evaluate a planning decision.

Examples:
- Planners assumed employment would grow in a corridor.
- An agency expected most passengers to access a station
  by car.
- A transportation model assumed a particular population
  growth rate.

Do NOT classify general background information or project
objectives as assumptions.

2. alternative

A specific, identifiable course of action considered
by the organization.

Examples:
- A proposed rail alignment.
- A No Build Alternative.
- A tunnel versus an elevated alignment.
- Two different station locations.

An alternative does not need to have been selected.

Do NOT classify generic descriptions of the study area
as alternatives.

3. constraint

An explicitly documented factor that limits the
organization's available choices or their feasibility.

Examples:
- Construction costs exceeding available funding.
- Right-of-way restrictions.
- Engineering limitations.
- Legal or regulatory requirements.
- Conflicts with existing infrastructure.

Do NOT classify geographic descriptions or design
characteristics as constraints unless the document
explicitly identifies them as limitations.

4. forecast

An explicitly stated prediction about future conditions
or outcomes.

Examples:
- Projected daily ridership in 2030.
- Expected population growth.
- Forecast travel-time savings.
- Estimated future operating costs.

Do NOT classify project intentions, objectives,
or descriptions of future decision procedures as forecasts.

5. evaluation_criterion

A dimension or standard explicitly used, or proposed
to be used, to compare planning alternatives.

Examples:
- Capital and operating cost.
- Projected ridership.
- Environmental impact.
- Travel time.
- Construction disruption.
- Accessibility.

Do NOT infer evaluation criteria merely because a
subject is mentioned in the document.

6. decision_rationale

An explicitly documented reason why an organization
selected, preferred, rejected, modified, or eliminated
a particular alternative.

Examples:
- Route A was rejected because of excessive costs.
- Route B was preferred because it served more jobs.
- An alignment was revised to avoid utility conflicts.

A general statement of project purpose is NOT a
decision rationale unless it is explicitly connected
to choosing or rejecting an alternative.

7. project_objective

An explicitly stated goal, need, or problem that
motivates the planning project.

Examples:
- Increase rail capacity.
- Reduce congestion.
- Improve transit accessibility.
- Maintain safe and reliable service.
- Address projected transportation demand.

Project objectives describe what the organization
seeks to accomplish, not necessarily why it prefers
one alternative over another.

CRITICAL EXTRACTION RULES
-------------------------

1. THE CATEGORIES ARE NOT A CHECKLIST.

The seven categories are possible labels, not seven
required outputs.

A passage may contain findings from only one category,
several categories, or none.

NEVER generate a finding merely to populate a category.

An empty findings list is a valid and desirable result
when no relevant evidence exists.

2. DO NOT INFER HIDDEN MOTIVATIONS.

Extract documented organizational statements,
not your interpretation of what decision-makers
were privately thinking.

Distinguish what the document asserts from what
might actually have motivated the decision.

3. EVERY FINDING MUST BE GROUNDED.

For every finding:
- Provide one concise, specific claim.
- Include an exact quotation from the supplied text.
- Ensure the quotation directly supports the claim.
- Do not invent quotations, numbers, or dates.
- Do not strengthen or generalize the source's wording.

If a claim cannot be supported with a specific quotation,
do not extract it.

4. AVOID DUPLICATE FINDINGS.

Do not repeatedly extract the same substantive claim
using different wording.

Do not reuse one quotation across multiple categories
unless it independently and explicitly supports
distinct claims belonging to those categories.

5. PRESERVE HISTORICAL CONTEXT.

Do not evaluate past decisions using present-day knowledge.

Do not assume that a forecast was accurate or inaccurate.

Do not describe an alternative as selected or rejected
unless the supplied text establishes that status.

Do not attribute statements to a particular agency
unless the document supports that attribution.

6. DISTINGUISH FACTS FROM RESEARCH CONSTRUCTS.

A source passage can be factually accurate but still
fail to qualify for a research category.

For example:

"The study corridor extends 2.5 miles through downtown."

This is geographic context, not necessarily a constraint.

"The agency will select a preferred alternative."

This is a procedural statement, not a forecast.

"The build alternatives provide additional capacity."

This may establish a project objective or anticipated
benefit, but does not establish a decision rationale
for preferring one specific alternative.

7. USE CONSERVATIVE CLASSIFICATION.

If a statement does not clearly satisfy a category
definition, omit it.

Prefer fewer, well-supported findings over many
speculative or weakly classified findings.

8. EXPLAIN EACH CLASSIFICATION.

For every finding, provide a short explanation of why
the evidence meets the assigned category definition.

The explanation must refer to the meaning of the
quoted passage, not merely repeat the category name.

9. KEEP CLAIMS ATOMIC.

Each finding should express one independently
verifiable proposition.

If a passage contains multiple independent claims,
extract them separately when they are substantively
relevant.

10. DO NOT INVENT MISSING INFORMATION.

If the supplied passage does not establish:
- the decision date,
- the responsible organization,
- whether an alternative was selected,
- the reasons for a decision,
- or a numerical forecast,

do not invent or assume that information.

OUTPUT REQUIREMENTS
-------------------

Return structured data matching the provided JSON schema.

Each finding must contain:
- category
- claim
- evidence_quote
- classification_reason

The findings list can be empty.

Do not include generic summaries, introductory
commentary, or conclusions.

Your primary goal is research-grade evidence
extraction, not maximizing the number of findings.
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
