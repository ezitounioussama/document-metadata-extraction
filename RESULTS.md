# Results — Structured Metadata from a Batch of Documents

Captured from a real run. `qwen3:8b` via `ChatOllama(reasoning=False)`, local Ollama, `temperature=0`.
Raw log: [`docs/output.txt`](docs/output.txt) · Dataset: [`output/`](output/)

```bash
ollama serve
python3 extract.py
python3 tests.py     # 24 tests, no model needed
```

Pipeline: **load → `PromptTemplate | ChatOllama | JsonOutputParser` → Pydantic → grounding → JSON + CSV**

---

## Input: the batch

Six documents, four formats, deliberately including awkward cases:

| File | Format | Chars | Why it is in the batch |
|---|---|---|---|
| `ai_trends_2025.txt` | txt | 940 | Clean case — title, author, ISO date all stated |
| `climate_agriculture.html` | html | 704 | Raw HTML; author and date also in `<meta>` tags |
| `ocr_scan_notes.txt` | txt | 634 | Simulated OCR: noise (`spi11`), date as `15/09/2024`, **no author** |
| `press_release_undated.txt` | txt | 512 | **No author, no date at all** |
| `product_memo.md` | md | 740 | Markdown, author in bold, **no date** |
| `vendor_security_assessment.docx` | docx | 952 | Real `.docx` (zip of XML), read without python-docx |

---

## Output: the structured dataset

All six rows validated. `output/metadata.csv`:

| source_file | title | author | publication_date | document_type |
|---|---|---|---|---|
| ai_trends_2025.txt | AI Trends 2025 | Dr. Sarah Lee | 2025-05-01 | report |
| climate_agriculture.html | Climate Change and Agriculture | John Smith | 2024-09-15 | article |
| ocr_scan_notes.txt | QUARTERLY SAFETY REVIEW -- INTERNAL | *(null)* | 2024-09-15 | report |
| press_release_undated.txt | Northwind Logistics Opens Automated Sorting Facility | *(null)* | *(null)* | press_release |
| product_memo.md | Internal Memo: Search Relevance Regression | Priya Raman | *(null)* | memo |
| vendor_security_assessment.docx | Vendor Security Assessment: CloudVault | Miguel Torres, Security Architect | 2025-02-18 | assessment |

One JSON record:

```json
{
  "source_file": "climate_agriculture.html",
  "format": "html",
  "title": "Climate Change and Agriculture",
  "author": "John Smith",
  "publication_date": "2024-09-15",
  "keywords": ["climate change", "agriculture", "sustainable farming",
               "drought-resistant", "no-till", "policy gaps"],
  "document_type": "article"
}
```

```
documents processed : 6
valid               : 6
rejected            : 0
with no author      : 2
with no date        : 2
```

Every non-null value in that table is present in its source document. Two dates and two authors
are null because those documents genuinely do not state them — which is the correct answer, not a
gap.

---

## Three problems found while building this

### 1. `JsonOutputParser` does not validate. At all.

The brief's section is called "Validation with Pydantic", but the script never validates. Passing
`pydantic_object=DocumentMetadata` only builds the format instructions. Verified directly:

```python
parser = JsonOutputParser(pydantic_object=DocumentMetadata)
parser.parse('{"title": 123, "keywords": "not-a-list", "document_type": "invented"}')
# -> {'title': 123, 'keywords': 'not-a-list', 'document_type': 'invented'}
#    accepted. no error.
```

Every one of those violates the schema. Validation happens only on the explicit call:

```python
DocumentMetadata(**parsed)   # ValidationError: 3 errors
```

That call is the missing line. There is a test asserting the parser accepts schema-violating JSON,
so the distinction cannot quietly regress.

**Related:** the brief's `metadata.dict()` raises `AttributeError`. `parse()` returns a plain
`dict`, and a dict has no `.dict()` method.

### 2. My own prompt example leaked into the data

First run, `product_memo.md` came back with `publication_date: 2024-09-15`. The memo contains no
date anywhere. That value came from my prompt's own instruction:

```
- publication_date must be YYYY-MM-DD. Convert other formats: 15/09/2024 becomes
  2024-09-15.                                                 ^^^^^^^^^^
```

The model read the illustration as data. Rewriting the rule abstractly —
*"reordering the parts, so DD/MM/YYYY becomes YYYY-MM-DD"* — stopped that specific copy.

Same failure class as a few-shot example leaking a figure: **a concrete value in a prompt
instruction can be read as content.** Keep instructions abstract when the field is factual.

### 3. Pydantic cannot catch a well-formed lie

With the leak fixed, the memo's date became `2023-04-01` — still invented, just no longer copied.
And the schema **accepted** it, correctly: `2023-04-01` is a valid ISO date. The constraint checks
format, and the format was perfect.

Shape validation and truth are different problems, so they need different code. `grounding.py`
checks extracted values against the document they came from:

```
publication_date '2023-04-01' does not appear in the source — dropped as unverified
```

The check has to be generous about formatting, because the model is *allowed* to reformat. It
generates every plausible written form of the same day before deciding:

| Source text | Extracted | Verdict |
|---|---|---|
| `Date of issue : 15/09/2024` | `2024-09-15` | **kept** — same day, reformatted |
| `published 15 September 2024` | `2024-09-15` | **kept** |
| no date anywhere | `2023-04-01` | **dropped** — invented |

Authors are matched on a name token rather than the whole string, because a model may return
`"Priya Raman, Search Platform"` where the document says `Priya Raman` — an exact match would
reject a correct answer. (`llama3.2:3b` did exactly that; `qwen3:8b` returns the bare name, and
still returns `"Miguel Torres, Security Architect"` for the docx, so the token match is still
what keeps that row.)

---

## What the format told us for free

The HTML document carries its metadata explicitly:

```html
<meta name="author" content="John Smith">
<meta name="date" content="2024-09-15">
```

The model **missed the author** on that document and returned null, despite the byline appearing
twice. Rather than accept the loss, the loader keeps `<title>` and `<meta>` values as hints and the
pipeline refills nulls from them:

```
[valid] climate_agriculture.html
    author : John Smith
    ! author recovered from HTML meta tag: 'John Smith'
```

A value the file format states outright is better evidence than one a 3B model infers from prose.
Order matters: verification runs **before** hint recovery, so a value the format declared is never
dropped for failing a source check.

---

## The full pipeline, in order

| Step | Does what | Catches |
|---|---|---|
| `loaders.py` | txt / md / html / docx → text + format hints | unreadable files, unsupported formats |
| `PromptTemplate \| ChatOllama \| JsonOutputParser` | extraction → dict | malformed JSON |
| `DocumentMetadata(**parsed)` | **schema validation** | wrong types, bad date format, invalid document_type, too few keywords |
| repair pass | feeds validation errors back for one retry | recoverable single-field mistakes |
| `grounding.py` | verify against source, recover from hints | **hallucinated but well-formed values** |
| re-validate | after grounding edited the values | edits that broke the schema |
| export | JSON + CSV + `rejected.json` | — |

Rejected rows are written to `output/rejected.json` rather than dropped. A row that failed
validation is a data-quality signal; discarding it silently hides how much of the batch was lost.

---

## Normalisation the schema does quietly

| Model returned | Stored | Why |
|---|---|---|
| `["AI", " ai ", "Ethics"]` | `["ai", "ethics"]` | lowercased, trimmed, deduplicated — otherwise grouping the dataset is impossible |
| `"Unknown"`, `"N/A"`, `"null"`, `""` | `None` | four spellings of "missing" become one |
| `"null"` as a date | `None` | otherwise stored as a 4-character string that passes no date check |
| `document_type: "technical report"` | **rejected** | closed `Literal` set, so `report`/`Report`/`technical report` cannot become three categories |

---

## Tests

```
$ python3 tests.py
Ran 24 tests in 0.002s

OK
```

Covering: the parser-does-not-validate trap, every schema constraint including impossible dates
like `2024-13-45`, keyword normalisation, placeholder-author collapsing, date grounding across five
written formats, author token matching, hint recovery precedence, and all four loaders.

One test failed while writing it and was my mistake, not the code's: I checked that a verified date
produced no warnings, but the source string I passed did not contain the author — so grounding
correctly flagged the author. The check was working; the test was wrong.

---

Author: **Oussama Ezitouni**
