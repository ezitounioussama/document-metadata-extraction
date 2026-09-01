# Structured Metadata Extraction from a Batch of Documents

Turn a folder of unstructured documents — txt, md, html, docx — into a validated, structured
dataset. Six documents in, six rows out, with every non-null value verified against the source it
came from.

Re-run on `qwen3:8b` (it was built on `llama3.2:3b`), the dataset is the same except two rows read
better: the memo's author comes back as the bare `Priya Raman` rather than `Priya Raman, Search
Platform`, and the OCR document is classified `report` instead of `memo`.

The verification is there because of what happened without it. `JsonOutputParser` looks like it
validates when you pass it `pydantic_object=...`, and it does not: it returns a plain dict and
will happily hand back `{"title": 123, "keywords": "not-a-list"}`. Adding real Pydantic validation
fixed the shape, and then the model invented `2023-04-01` for a document with no date — a
perfectly valid ISO date, so the schema accepted it. Shape and truth are different checks, so
`grounding.py` looks for each extracted date and author in its own source document and drops
anything that isn't there. The nulls in the output are correct answers.

```bash
ollama serve && ollama pull qwen3:8b
uv venv
uv pip install langchain-core langchain-ollama pydantic

.venv/bin/python extract.py       # run the pipeline on data/
.venv/bin/python tests.py         # 24 tests, no model needed
```

`ChatOllama` replaces `ChatOpenAI` from the brief — no API key here, and a chat model is
interchangeable in the chain.

## Also in this repo

- **[RESULTS.md](RESULTS.md)** — the extracted dataset, the full pipeline in order, and the three
  problems found while building (one of them in the brief's own example script)
- `output/` — `metadata.json`, `metadata.csv`, `rejected.json`

---

Author: **Oussama Ezitouni**
