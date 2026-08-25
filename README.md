# Structured Metadata Extraction from a Batch of Documents

Turn a folder of unstructured documents into a validated, structured dataset.

> **[RESULTS.md](RESULTS.md)** — the extracted dataset, and the three problems found while
> building it (including one in the brief's own example script).

**load → `PromptTemplate | ChatOllama | JsonOutputParser` → Pydantic → grounding → JSON + CSV**

```bash
ollama serve && ollama pull llama3.2:3b
python3 -m venv .venv
.venv/bin/python -m pip install langchain-core langchain-ollama pydantic

.venv/bin/python extract.py       # run the pipeline on data/
.venv/bin/python tests.py         # 24 tests, no model needed
```

| File | Contents |
|---|---|
| `extract.py` | The pipeline: prompt, chain, validation, repair pass, export |
| `schema.py` | The Pydantic model — where validation actually happens |
| `loaders.py` | txt / md / html / docx readers, standard library only |
| `grounding.py` | Checks extracted values against the source text |
| `tests.py` | 24 tests |
| `data/` | Six input documents, four formats |
| `output/` | `metadata.json`, `metadata.csv`, `rejected.json` |

`ChatOllama` replaces `ChatOpenAI` from the brief — no API key here, and a chat model is
interchangeable in the chain.

## Result

Six documents in, six validated rows out, every non-null value verified against its source:

| source_file | author | publication_date | document_type |
|---|---|---|---|
| ai_trends_2025.txt | Dr. Sarah Lee | 2025-05-01 | report |
| climate_agriculture.html | John Smith | 2024-09-15 | article |
| ocr_scan_notes.txt | *(null)* | 2024-09-15 | memo |
| press_release_undated.txt | *(null)* | *(null)* | press_release |
| product_memo.md | Priya Raman, Search Platform | *(null)* | memo |
| vendor_security_assessment.docx | Miguel Torres, Security Architect | 2025-02-18 | assessment |

The nulls are correct answers: those documents state no author or no date. The OCR document's
`15/09/2024` was converted to ISO and verified as the same day.

## Three findings

**`JsonOutputParser` does not validate.** Passing `pydantic_object=...` only builds the format
instructions. `parse()` returns a plain dict and accepts `{"title": 123, "keywords": "not-a-list"}`
without complaint. Validation happens only on `DocumentMetadata(**parsed)` — the line the brief's
example is missing. (Its `metadata.dict()` also raises `AttributeError`, since `parse()` returns a
dict.)

**A concrete value in a prompt instruction can be read as data.** My rule said *"15/09/2024 becomes
2024-09-15"*, and the model stamped `2024-09-15` onto a memo containing no date. Rewriting the rule
abstractly stopped it.

**Pydantic cannot catch a well-formed lie.** The invented date then became `2023-04-01` — a valid
ISO date, so the schema accepted it. Shape and truth are different checks: `grounding.py` verifies
each date and author actually appears in its source document, and drops it if not. It is generous
about formatting, so a real date written `15 September 2024` still passes as `2024-09-15`.

---

Author: **Oussama Ezitouni**
