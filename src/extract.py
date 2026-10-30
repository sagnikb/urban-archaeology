
import json
import hashlib
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from importlib.metadata import PackageNotFoundError, version
from typing import Literal

import pymupdf as fitz
from ollama import chat
from ollama import Client
from pydantic import BaseModel


MODEL = "qwen3:4b"
SEED = 42


# ============================================================
# OUTPUT SCHEMA
# ============================================================

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
    epistemic_status: Literal["stated fact", "assumption", "projection", "procedure"]
    attributed_to: str
    refers_to_alternative: str


class Extraction(BaseModel):
    findings: list[Finding]


# ============================================================
# SYSTEM PROMPT
# ============================================================

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

Do NOT classify general background information or
project objectives as assumptions.

2. alternative

A specific, identifiable course of action considered
by the organization.

Examples:
- A proposed rail alignment.
- A No Build Alternative.
- A tunnel versus an elevated alignment.
- Two different station locations.

Do NOT classify generic descriptions of the study area
as alternatives.

SPECIAL RULES FOR ALTERNATIVES:

Distinguish between:

A. Named or independently identifiable alternatives:
   Example: "No Build Alternative" or "Alignment B".

B. Groups of alternatives:
   Example: "Four build alternatives were considered."

C. Design attributes:
   Example: "The alignments include underground stations."

Only A qualifies as an alternative finding.

For B, do not invent the identities of alternatives.

For C, do not create alternative findings unless the
document explicitly treats the configuration as a
separate selectable option.

Do not count umbrella descriptions or subsets of
previously extracted alternatives as new options.

3. constraint

An explicitly documented factor that limits the
organization's available choices or their feasibility.

Examples:
- Funding limitations.
- Right-of-way restrictions.
- Engineering limitations.
- Legal requirements.
- Conflicts with existing infrastructure.

Geographic descriptions and design characteristics
are not automatically constraints.

4. forecast

An explicitly stated prediction about future conditions
or outcomes.

Examples:
- Projected daily ridership in 2030.
- Expected population growth.
- Forecast travel-time savings.
- Estimated future operating costs.

Project intentions and decision procedures are NOT forecasts.

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

Do not infer criteria merely because a topic is mentioned.

6. decision_rationale

An explicitly documented reason why an organization
selected, preferred, rejected, modified, or eliminated
a particular alternative.

A general project objective is NOT sufficient unless
explicitly connected to a choice between alternatives.

7. project_objective

An explicitly stated goal, need, or problem that
motivates the planning project.

Examples:
- Increase rail capacity.
- Reduce congestion.
- Improve accessibility.
- Maintain reliable service.
- Address projected transportation demand.

Project objectives describe what the organization
seeks to accomplish, not necessarily why it prefers
one alternative over another.

CRITICAL EXTRACTION RULES
-------------------------

1. CATEGORIES ARE NOT A CHECKLIST.

The seven categories are possible labels, not seven
required outputs.

A passage may contain findings from only one category,
several categories, or none.

Never generate a finding merely to populate a category.

An empty findings list is valid.

2. DO NOT INFER HIDDEN MOTIVATIONS.

Extract documented organizational statements, not
speculation about what decision-makers privately believed.

3. EVERY FINDING MUST BE GROUNDED.

For each finding:
- Provide a concise, specific claim.
- Copy an exact supporting quotation.
- Ensure the quotation directly supports the claim.
- Do not invent quotations, numbers, or dates.
- Do not exaggerate the source's wording.

If a claim cannot be supported, omit it.

4. AVOID DUPLICATES.

Do not repeatedly extract the same substantive claim.

Do not reuse one quotation across multiple categories
unless it explicitly supports distinct claims.

5. PRESERVE HISTORICAL CONTEXT.

Do not evaluate past decisions using present-day knowledge.

Do not assume forecasts were accurate or inaccurate.

Do not describe an alternative as selected or rejected
unless the supplied text establishes that status.

6. DISTINGUISH FACTS FROM RESEARCH CONSTRUCTS.

"The corridor extends 2.5 miles through downtown"
is geographic context, not necessarily a constraint.

"The agency will select a preferred alternative"
is a procedural statement, not a forecast.

"The build alternatives provide additional capacity"
may establish a project objective but does not establish
a rationale for preferring one specific alternative.

7. USE CONSERVATIVE CLASSIFICATION.

Prefer fewer, well-supported findings over many weakly
classified findings.

8. EXPLAIN CLASSIFICATIONS.

For every finding, briefly explain why the supporting
passage qualifies for its category.

9. KEEP CLAIMS ATOMIC.

Each finding should express one independently
verifiable proposition.

10. DO NOT INVENT MISSING INFORMATION.

Never invent dates, agencies, outcomes, alternative
names, decision rationales, or numerical forecasts.

CONTEXT HANDLING
----------------

You may receive surrounding pages as additional context.

Use surrounding pages ONLY to interpret the target page.

Extract findings ONLY from the target page.

Every evidence_quote must appear verbatim in the target
page, not merely in the surrounding context.

Do not extract findings exclusively supported by
surrounding pages.

Do not allow background context to override what the
target page actually states.

OUTPUT REQUIREMENTS
-------------------

Return structured data matching the provided JSON schema.

Each finding must contain:
- category
- claim
- evidence_quote
- classification_reason
- epistemic_status: stated fact, assumption, projection, or procedure
- attributed_to: named source/organization, or empty if none is stated
- refers_to_alternative: explicit alternative identifier, or empty if none

The findings list may be empty.

Do not include commentary outside the JSON.
"""


# ============================================================
# UTILITIES
# ============================================================

def normalize(text):
    """Normalize whitespace for quotation comparison."""
    return re.sub(r"\s+", " ", text).strip()


def get_page_text(pdf, index):
    """Get text from a PDF page, zero-indexed."""
    return pdf[index].get_text(sort=True)


def build_context(
    pdf,
    target_index,
    context_pages=1,
    context_chars=2000,
):
    """
    # More faithful attribution/alternative labels are expected only when
    # the source explicitly names them; never create labels for convenience.
    Retrieve neighboring PDF pages.

    Does not include the target page.
    """
    parts = []

    start = max(0, target_index - context_pages)
    end = min(len(pdf), target_index + context_pages + 1)

    for j in range(start, end):
        if j == target_index:
            continue

        text = get_page_text(pdf, j)

        parts.append(
            f"PDF PAGE {j + 1}:\n"
            f"{text[:context_chars]}"
        )

    return "\n\n".join(parts)


# ============================================================
# LLM EXTRACTION
# ============================================================

def analyze(
    target_text,
    filename,
    page,
    printed_page_label,
    seed=SEED,
    parse_mode="single",
    context_text="",
):
    """
    Analyze one target page.

    single:
        Model receives only target page.

    context:
        Model receives neighboring pages as context,
        but findings must be grounded in the target.
    """

    if parse_mode == "single":
        user_prompt = (
            f"TARGET PDF PAGE {page}:\n\n"
            f"{target_text}\n\n"
            "Extract findings only from this page."
        )

    elif parse_mode == "context":
        user_prompt = (
            "SURROUNDING CONTEXT — FOR INTERPRETATION ONLY:\n"
            "Do not extract findings from this section.\n\n"
            f"{context_text}\n\n"
            "========================================\n"
            f"TARGET PDF PAGE {page}:\n"
            "========================================\n\n"
            f"{target_text}\n\n"
            "Extract findings ONLY from the TARGET PDF PAGE. "
            "Every evidence_quote must occur in the target page."
        )

    else:
        raise ValueError(
            "parse_mode must be 'single' or 'context'"
        )

    response = chat(
        model=MODEL,
        think=False,
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],
        format=Extraction.model_json_schema(),
        options={
            "temperature": 0,
            "seed": seed,
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

        # Verify against target text, not context.
        verified = (
            bool(quote)
            and quote in normalize(target_text)
        )

        record["document"] = filename
        record["page"] = page
        record["printed_page_label"] = printed_page_label
        record["finding_id"] = stable_finding_id(
            filename, page, record["category"], record["evidence_quote"]
        )
        record["quote_verified"] = verified
        record["parse_mode"] = parse_mode

        records.append(record)

    return records


def stable_finding_id(filename, page, category, quote):
    import hashlib
    key = "|".join((filename, str(page), category, normalize(quote)))
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]


# ============================================================
# MAIN PIPELINE
# ============================================================

def main(
    doc_path="./documents/d2deismarch2010.pdf",
    start_page=5,
    pages=1,
    parse_mode="single",
    context_pages=1,
    max_chars=5000,
    context_chars=2000,
    run_name=None,
    seed=SEED,
):
    """
    Run PDF extraction directly from VS Code.

    start_page:
        1-indexed PDF page number.

    pages:
        Number of target pages to process.

    parse_mode:
        'single' or 'context'.

    context_pages:
        Number of neighboring pages on EACH side.

    max_chars:
        Maximum target-page text length.

    context_chars:
        Maximum text length from each neighboring page.
    """

    if parse_mode not in ("single", "context"):
        raise ValueError(
            "parse_mode must be 'single' or 'context'"
        )

    path = Path(doc_path)

    if not path.exists():
        raise FileNotFoundError(path)

    run_name = run_name or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = Path(f"outputs/findings_{parse_mode}_{run_name}.json")
    verified_output = Path(f"outputs/findings_{parse_mode}_{run_name}_verified.json")
    error_output = Path(f"outputs/run_error_{parse_mode}_{run_name}.json")
    output.parent.mkdir(parents=True, exist_ok=True)

    results = []
    try:
        ollama_version = version("ollama")
    except PackageNotFoundError:
        ollama_version = "unknown"
    try:
        ollama_server_version = Client().version().get("version", "unavailable")
    except Exception:
        ollama_server_version = "unavailable"
    try:
        git_commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except Exception:
        git_commit = "unavailable"
    run_details = {
        "run_id": run_name,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "pdf": path.name,
        "pdf_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "model": MODEL,
        "ollama_python_version": ollama_version,
        "ollama_server_version": ollama_server_version,
        "prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest(),
        "git_commit": git_commit,
        "settings": {"start_page": start_page, "pages": pages, "parse_mode": parse_mode,
                     "context_pages": context_pages, "max_chars": max_chars,
                     "context_chars": context_chars, "temperature": 0, "seed": seed,
                     "num_ctx": 8192},
    }
    errors = []

    with fitz.open(path) as pdf:
        start = start_page - 1
        end = min(start + pages, len(pdf))

        if start < 0 or start >= len(pdf):
            raise ValueError("Start page is outside the PDF")

        for i in range(start, end):
            page = i + 1
            target_text = get_page_text(pdf, i)
            footer = " ".join(page_word[4] for page_word in pdf[i].get_text("words")
                              if page_word[1] > pdf[i].rect.height * 0.88)
            label_match = re.search(r"\bPage\s+(\d+-\d+)\b", footer, re.IGNORECASE)
            printed_page_label = label_match.group(1) if label_match else (pdf[i].get_label() or "")
            target_text = target_text[:max_chars]

            print(
                f"Processing PDF page {page} "
                f"[{parse_mode}]...",
                flush=True,
            )

            if len(target_text.strip()) < 100:
                print("  Skipped: insufficient text")
                continue

            context_text = ""

            if parse_mode == "context":
                context_text = build_context(
                    pdf,
                    target_index=i,
                    context_pages=context_pages,
                    context_chars=context_chars,
                )

            try:
                records = analyze(
                    target_text=target_text,
                    filename=path.name,
                    page=page,
                    printed_page_label=printed_page_label,
                    seed=seed,
                    parse_mode=parse_mode,
                    context_text=context_text,
                )

                results.extend(records)

                verified = sum(
                    item["quote_verified"]
                    for item in records
                )

                print(
                    f"  Found {len(records)} findings "
                    f"({verified} quotations verified)"
                )

            except Exception as error:
                print(f"  Error on page {page}: {error}")
                errors.append({"page": page, "error": str(error)})
                error_output.write_text(
                    json.dumps({"run_details": run_details, "status": "failed", "errors": errors},
                               indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
                raise RuntimeError(f"Extraction failed on PDF page {page}; details saved to {error_output}") from error

    # Publish normal outputs only after every requested page succeeded.
    output.write_text(
        json.dumps({"run_details": run_details, "findings": results, "errors": [], "status": "complete"},
                   indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    verified_output.write_text(
        json.dumps({"run_details": run_details,
                    "findings": [item for item in results if item["quote_verified"]],
                    "errors": [], "status": "complete"}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"\nTotal findings: {len(results)}")
    print(f"Saved to: {output}")
    print(f"Saved verified findings to: {verified_output}")

    return results


# ============================================================
# RUN CONFIGURATION — EDIT THIS IN VS CODE
# ============================================================

if __name__ == "__main__":
    main(
        doc_path="./documents/d2deismarch2010.pdf",
        start_page=91,
        pages=1,
        parse_mode="single",  # "single" or "context"
        context_pages=1,
        max_chars=5000,
        context_chars=2000,
    )
