"""Load a batch of mixed-format documents into plain text.

The brief's input is "PDFs, text files, scanned reports, or raw HTML pages", so the
loader has to cope with more than one format. Standard library only — no PDF
dependency is pulled in for a format this batch does not contain, and a stub
raises a clear message if one appears.

Each loader returns the document's text plus a note about where the text came
from, because "the author was in an HTML meta tag" is useful context when a later
extraction goes wrong.
"""

import html
import re
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import List


@dataclass
class Document:
    """One loaded document, ready for extraction."""

    path: Path
    text: str
    fmt: str
    hints: dict = field(default_factory=dict)   # anything the format told us for free

    @property
    def name(self) -> str:
        return self.path.name

    @property
    def chars(self) -> int:
        return len(self.text)


def load_text(path: Path) -> Document:
    """.txt and .md — already plain text."""
    return Document(path=path, text=path.read_text(encoding="utf-8").strip(),
                    fmt=path.suffix.lstrip("."))


def load_html(path: Path) -> Document:
    """Strip tags, but keep what the <head> already told us.

    HTML often carries the metadata explicitly in `<title>` and
    `<meta name="author">`. Those are collected as hints rather than thrown away —
    a value the format states outright is more trustworthy than one the model
    infers from prose, and it gives something to check the model against.
    """
    raw = path.read_text(encoding="utf-8")

    hints = {}
    title = re.search(r"<title[^>]*>(.*?)</title>", raw, re.S | re.I)
    if title:
        hints["title"] = html.unescape(title.group(1).strip())

    for name in ("author", "date"):
        meta = re.search(
            rf'<meta\s+name=["\']{name}["\']\s+content=["\'](.*?)["\']', raw, re.I
        )
        if meta:
            hints[name] = html.unescape(meta.group(1).strip())

    # Drop script/style bodies before stripping tags, or their contents become text.
    body = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", raw, flags=re.S | re.I)
    body = re.sub(r"<[^>]+>", " ", body)
    body = html.unescape(body)
    body = re.sub(r"[ \t]+", " ", body)
    body = re.sub(r"\n\s*\n\s*\n+", "\n\n", body).strip()

    return Document(path=path, text=body, fmt="html", hints=hints)


def load_docx(path: Path) -> Document:
    """A .docx is a zip of XML; the body lives in word/document.xml.

    Paragraph text sits in <w:t> runs, and Word splits a sentence across several
    runs whenever formatting changes, so the runs of each paragraph are rejoined.
    """
    with zipfile.ZipFile(path) as archive:
        if "word/document.xml" not in archive.namelist():
            raise ValueError(f"{path.name} has no word/document.xml — not a .docx?")
        xml = archive.read("word/document.xml").decode("utf-8")

    paragraphs = []
    for block in re.findall(r"<w:p[ >].*?</w:p>", xml, re.S):
        runs = "".join(re.findall(r"<w:t[^>]*>(.*?)</w:t>", block, re.S))
        text = html.unescape(re.sub(r"<[^>]+>", "", runs)).strip()
        if text:
            paragraphs.append(text)

    return Document(path=path, text="\n\n".join(paragraphs), fmt="docx")


def load_pdf(path: Path) -> Document:
    """Not supported here, and saying so beats a confusing failure.

    This batch contains no PDFs. Adding pypdf just to have the branch would mean
    an untested dependency, so the stub names the fix instead.
    """
    raise NotImplementedError(
        f"{path.name}: PDF support needs an extra library. "
        "Install pypdf and extract with PdfReader(path).pages[i].extract_text()."
    )


LOADERS = {
    ".txt": load_text,
    ".md": load_text,
    ".html": load_html,
    ".htm": load_html,
    ".docx": load_docx,
    ".pdf": load_pdf,
}


def load_batch(directory) -> List[Document]:
    """Load every supported file in a directory, sorted for reproducibility."""
    directory = Path(directory)
    documents = []

    for path in sorted(directory.iterdir()):
        if not path.is_file():
            continue

        loader = LOADERS.get(path.suffix.lower())
        if loader is None:
            print(f"  skipped {path.name} (no loader for {path.suffix})")
            continue

        try:
            documents.append(loader(path))
        except (NotImplementedError, ValueError) as error:
            print(f"  skipped {path.name}: {error}")

    return documents
