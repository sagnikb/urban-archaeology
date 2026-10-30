"""Create a PDF copy with extracted finding quotations highlighted."""

import argparse
import json
import re
from pathlib import Path

import pymupdf


def search_prefix(quote, word_count=8):
    words = re.findall(r"\S+", quote)
    return " ".join(words[:word_count])


def create_audit_pdf(pdf_path, findings_path, output_path, target_page=91, single_page=None):
    data = json.loads(findings_path.read_text(encoding="utf-8"))
    findings = data.get("findings", []) if isinstance(data, dict) else data
    document = pymupdf.open(pdf_path)
    highlighted = 0
    missing = []
    for finding in findings:
        page_number = finding.get("page")
        quote = finding.get("evidence_quote", "")
        if page_number != target_page or not quote:
            continue
        page = document[page_number - 1]
        prefix = search_prefix(quote)
        rects = page.search_for(prefix)
        if not rects:
            missing.append(finding.get("finding_id") or quote[:60])
            continue
        annotation = page.add_highlight_annot(rects)
        annotation.set_info(title="Extracted finding", content=finding.get("finding_id", ""))
        annotation.update()
        highlighted += 1
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if single_page is None:
        document.save(output_path)
    else:
        if single_page < 1 or single_page > len(document):
            document.close()
            raise ValueError(f"Single page must be between 1 and {len(document)}")
        one_page = pymupdf.open()
        one_page.insert_pdf(document, from_page=single_page - 1, to_page=single_page - 1)
        one_page.save(output_path)
        one_page.close()
    document.close()
    print(f"Highlighted {highlighted} findings on PDF page {target_page}; saved {output_path}")
    if missing:
        print(f"Could not locate {len(missing)} quote prefixes: {', '.join(missing)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf", type=Path, default=Path("documents/d2deismarch2010.pdf"))
    parser.add_argument("--findings", type=Path, default=Path("outputs/findings_single_20261030T022825Z.json"))
    parser.add_argument("--output", type=Path, default=Path("outputs/page91_audit.pdf"))
    parser.add_argument("--page", type=int, default=91, help="1-indexed PDF page to audit")
    parser.add_argument("--single-page", type=int, default=None,
                        help="save only this 1-indexed PDF page in the output")
    args = parser.parse_args()
    create_audit_pdf(args.pdf, args.findings, args.output, args.page, args.single_page)


if __name__ == "__main__":
    main()
