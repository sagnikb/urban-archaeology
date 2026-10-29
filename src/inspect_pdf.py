
from pathlib import Path
import pymupdf as fitz

pdfs = list(Path("documents").glob("*.pdf"))

if not pdfs:
    raise FileNotFoundError("No PDFs in documents/")

for path in pdfs:
    with fitz.open(path) as doc:
        print(f"\nFile: {path.name}")
        print(f"Pages: {len(doc)}")
        for i in range(min(20, len(doc))):
            text = doc[i].get_text(sort=True)
            print(
                f"Page {i+1:3}: {len(text):5} characters | "
                f"{' '.join(text.split())[:90]}"
            )