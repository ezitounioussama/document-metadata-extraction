"""
Extract structured metadata from a batch of documents, validate it, export it.

    python3 extract.py                # runs the whole pipeline on data/
    python3 extract.py --no-retry     # skip the repair pass, to see raw failures

Pipeline:
    load  ->  PromptTemplate | ChatOllama | JsonOutputParser  ->  Pydantic  ->  JSON + CSV

`ChatOllama` stands in for `ChatOpenAI` from the brief: no API key here, and a
chat model is interchangeable in the chain either way.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from pydantic import ValidationError

from grounding import recover_from_hints, verify
from loaders import load_batch
from schema import DOCUMENT_TYPES, DocumentMetadata, ExtractionResult

MODEL = "llama3.2:3b"
DATA_DIR = Path("data")
OUT_DIR = Path("output")

# temperature=0: extraction is not creative, and a rerun should give the same
# dataset. Anything above 0 makes the output non-reproducible for no benefit.
model = ChatOllama(model=MODEL, temperature=0.0, num_predict=400)

parser = JsonOutputParser(pydantic_object=DocumentMetadata)

# The prompt does three jobs beyond naming the fields, each one added because the
# model got it wrong without it:
#   - spells out the closed list of document_type values
#   - says explicitly to use null rather than guess, or the model invents authors
#   - repeats the date format, because "May 2025" is otherwise very tempting
#
# The date rule is written abstractly (DD/MM/YYYY becomes YYYY-MM-DD) on purpose.
# An earlier version used a concrete example, "15/09/2024 becomes 2024-09-15", and
# the model copied that literal date into product_memo.md — a document containing
# no date at all. A concrete value in a prompt instruction can be read as data.
extract_prompt = PromptTemplate(
    template="""Extract metadata from the document text below.

Fields:
- title: the document's actual title
- author: the person who wrote it, or null if the document does not name one
- publication_date: the date in YYYY-MM-DD format, or null if no date is stated
- keywords: 2 to 10 lowercase topic keywords
- document_type: exactly one of {document_types}

Rules:
- Use null for author or publication_date if the document does not state them.
  Do not guess and do not invent a name or a date.
- publication_date must be YYYY-MM-DD. Convert other formats by reordering the
  parts, so DD/MM/YYYY becomes YYYY-MM-DD. If only a month or a year is given, or
  the document states no date at all, use null.
- Return JSON only.

{format_instructions}

Document text:
{document_text}
""",
    input_variables=["document_text"],
    partial_variables={
        "format_instructions": parser.get_format_instructions(),
        "document_types": ", ".join(DOCUMENT_TYPES),
    },
)

# The chain, composed with pipes. JsonOutputParser sits at the end so the chain
# hands back a dict — note that this is where the brief's `.dict()` call fails,
# because parse() returns a dict and a dict has no .dict().
extract_chain = extract_prompt | model | parser


REPAIR_PROMPT = PromptTemplate.from_template(
    """The JSON below failed validation. Fix ONLY the listed problems and return the
corrected JSON. Change nothing else.

Problems:
{errors}

Rules to satisfy:
- publication_date must be YYYY-MM-DD, or null if the document states no date
- keywords must be a list of 2 to 10 lowercase strings
- document_type must be exactly one of: {document_types}
- author must be a name, or null if unknown. Never "Unknown" or "N/A"

Invalid JSON:
{bad_json}

Return JSON only.
"""
)

repair_chain = REPAIR_PROMPT | model | parser


def describe_errors(error: ValidationError):
    """Turn a ValidationError into short lines a model can act on."""
    lines = []
    for item in error.errors():
        field = ".".join(str(part) for part in item["loc"]) or "(root)"
        lines.append(f"- {field}: {item['msg']}")
    return lines


def _ground(document, metadata, raw, attempts=1, notes=None) -> ExtractionResult:
    """Cross-check the validated metadata against the source text.

    Two passes, in this order:

      1. verify  — drop any date or author that does not appear in the document.
                   This is what catches a hallucinated but well-formed date, which
                   the schema cannot: "2023-04-01" is a valid ISO date.
      2. recover — refill nulls from metadata the file format stated outright,
                   such as an HTML <meta name="author"> tag.

    Verification runs first on purpose. Recovering from a hint and then verifying
    would risk dropping a value the format itself declared.
    """
    notes = list(notes or [])

    data, warnings = verify(metadata, document.text)
    notes.extend(warnings)

    data, recovered = recover_from_hints(data, document.hints)
    notes.extend(recovered)

    # Re-validate after editing, so the exported row is guaranteed to satisfy the
    # schema even though the values changed since the first check.
    try:
        grounded = DocumentMetadata(**data)
    except ValidationError as error:
        return ExtractionResult(
            source_file=document.name, format=document.fmt, status="invalid",
            errors=notes + describe_errors(error), raw_model_output=raw, attempts=attempts,
        )

    return ExtractionResult(
        source_file=document.name, format=document.fmt, status="valid",
        metadata=grounded, raw_model_output=raw, attempts=attempts, errors=notes,
    )


def extract_one(document, allow_retry=True) -> ExtractionResult:
    """Extract, validate, and repair once if validation fails."""
    raw = None

    try:
        parsed = extract_chain.invoke({"document_text": document.text})
        raw = json.dumps(parsed, ensure_ascii=False)
    except Exception as error:  # noqa: BLE001 - a malformed reply must not stop the batch
        return ExtractionResult(
            source_file=document.name,
            format=document.fmt,
            status="invalid",
            errors=[f"model or parser failure: {type(error).__name__}: {error}"],
        )

    # THIS is the validation step. JsonOutputParser did not do it.
    try:
        metadata = DocumentMetadata(**parsed)
        return _ground(document, metadata, raw, attempts=1)
    except ValidationError as first_error:
        problems = describe_errors(first_error)

    if not allow_retry:
        return ExtractionResult(
            source_file=document.name, format=document.fmt,
            status="invalid", errors=problems, raw_model_output=raw,
        )

    # One repair pass, with the validation errors fed back. Cheaper than discarding
    # the row, and the model usually only got one field wrong.
    try:
        repaired = repair_chain.invoke(
            {
                "errors": "\n".join(problems),
                "bad_json": raw,
                "document_types": ", ".join(DOCUMENT_TYPES),
            }
        )
        metadata = DocumentMetadata(**repaired)
        return _ground(
            document, metadata, json.dumps(repaired, ensure_ascii=False), attempts=2,
            notes=[f"repaired after: {p}" for p in problems],
        )
    except (ValidationError, Exception) as second_error:  # noqa: BLE001
        detail = (describe_errors(second_error)
                  if isinstance(second_error, ValidationError)
                  else [f"{type(second_error).__name__}: {second_error}"])
        return ExtractionResult(
            source_file=document.name, format=document.fmt, status="invalid",
            errors=problems + ["after repair:"] + detail,
            raw_model_output=raw, attempts=2,
        )


def export(results):
    """Write the clean dataset as JSON and CSV, and the rejects separately."""
    OUT_DIR.mkdir(exist_ok=True)

    valid = [r for r in results if r.status == "valid"]
    invalid = [r for r in results if r.status == "invalid"]

    # JSON: the full records, including which file each row came from.
    rows = [
        {"source_file": r.source_file, "format": r.format, **r.metadata.model_dump()}
        for r in valid
    ]
    (OUT_DIR / "metadata.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    # CSV: keywords joined, because a list does not fit a single cell.
    with open(OUT_DIR / "metadata.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["source_file", "format", "title", "author",
                         "publication_date", "keywords", "document_type"])
        for row in rows:
            writer.writerow([
                row["source_file"], row["format"], row["title"],
                row["author"] or "", row["publication_date"] or "",
                "; ".join(row["keywords"]), row["document_type"],
            ])

    # Rejects kept, not dropped: a row that failed validation is a data-quality
    # signal, and silently discarding it hides how much the batch lost.
    (OUT_DIR / "rejected.json").write_text(
        json.dumps(
            [{"source_file": r.source_file, "errors": r.errors,
              "raw_model_output": r.raw_model_output} for r in invalid],
            indent=2, ensure_ascii=False,
        ) + "\n",
        encoding="utf-8",
    )

    return rows, invalid


def main():
    arguments = argparse.ArgumentParser(description="Extract document metadata.")
    arguments.add_argument("--no-retry", action="store_true",
                           help="Do not attempt the repair pass on invalid output.")
    options = arguments.parse_args()

    line = "=" * 84
    print(f"{line}\nSETUP\n{line}")
    print(f"  Model    : {MODEL} via ChatOllama (local)")
    print(f"  Chain    : PromptTemplate | ChatOllama | JsonOutputParser -> Pydantic -> grounding")
    print(f"  Retry    : {'off' if options.no_retry else 'on (one repair pass)'}")

    print(f"\n{line}\nLOADING THE BATCH\n{line}")
    documents = load_batch(DATA_DIR)
    print(f"  {len(documents)} documents loaded\n")
    for document in documents:
        hint = f"  hints: {document.hints}" if document.hints else ""
        print(f"    {document.name:36} {document.fmt:5} {document.chars:5} chars{hint}")

    print(f"\n{line}\nEXTRACTING\n{line}")
    results = []
    for document in documents:
        result = extract_one(document, allow_retry=not options.no_retry)
        results.append(result)

        mark = "valid  " if result.status == "valid" else "INVALID"
        note = f" (repaired, {result.attempts} attempts)" if result.attempts > 1 and result.status == "valid" else ""
        print(f"\n  [{mark}] {result.source_file}{note}")

        if result.metadata:
            data = result.metadata
            print(f"      title            : {data.title}")
            print(f"      author           : {data.author}")
            print(f"      publication_date : {data.publication_date}")
            print(f"      keywords         : {', '.join(data.keywords)}")
            print(f"      document_type    : {data.document_type}")
        for message in result.errors:
            print(f"      ! {message}")

    print(f"\n{line}\nEXPORT\n{line}")
    rows, invalid = export(results)
    print(f"  output/metadata.json   {len(rows)} rows")
    print(f"  output/metadata.csv    {len(rows)} rows")
    print(f"  output/rejected.json   {len(invalid)} rows")

    print(f"\n{line}\nSUMMARY\n{line}")
    repaired = sum(1 for r in results if r.status == "valid" and r.attempts > 1)
    print(f"  documents processed : {len(results)}")
    print(f"  valid               : {len(rows)}")
    print(f"    of which repaired : {repaired}")
    print(f"  rejected            : {len(invalid)}")
    print(f"  with no author      : {sum(1 for r in rows if not r['author'])}")
    print(f"  with no date        : {sum(1 for r in rows if not r['publication_date'])}")

    print(f"\n{line}\nDone.\n{line}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
