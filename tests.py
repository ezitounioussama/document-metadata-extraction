"""
Tests for the schema, the loaders and the grounding checks.

    python3 tests.py

No model needed: validation, parsing and grounding are all pure functions. The
LLM call is the only part that needs Ollama, and it is not what breaks.
"""

import unittest
from pathlib import Path

from langchain_core.output_parsers import JsonOutputParser
from pydantic import ValidationError

from grounding import author_appears_in, date_appears_in, recover_from_hints, verify
from loaders import load_batch, load_docx, load_html, load_text
from schema import DocumentMetadata

DATA = Path("data")

VALID = {
    "title": "AI Trends 2025",
    "author": "Dr. Sarah Lee",
    "publication_date": "2025-05-01",
    "keywords": ["ai", "ethics"],
    "document_type": "report",
}


class TestJsonOutputParserDoesNotValidate(unittest.TestCase):
    """The trap in the brief's script, pinned down as a test.

    JsonOutputParser takes a pydantic_object, which reads like validation. It is
    not: the model is used to build the format instructions, and parse() returns a
    plain dict with no checking at all.
    """

    def setUp(self):
        self.parser = JsonOutputParser(pydantic_object=DocumentMetadata)

    def test_parse_returns_a_plain_dict(self):
        result = self.parser.parse('{"title": "X"}')

        self.assertIsInstance(result, dict)
        # The brief calls metadata.dict() on this, which raises AttributeError.
        self.assertFalse(hasattr(result, "dict"))

    def test_parser_accepts_output_that_violates_the_schema(self):
        bad = '{"title": 123, "keywords": "not-a-list", "document_type": "invented"}'

        result = self.parser.parse(bad)          # no error
        self.assertEqual(result["title"], 123)

        # Only the explicit model call rejects it.
        with self.assertRaises(ValidationError):
            DocumentMetadata(**result)


class TestSchemaValidation(unittest.TestCase):
    def test_a_good_record_validates(self):
        self.assertEqual(DocumentMetadata(**VALID).document_type, "report")

    def test_wrong_date_format_is_rejected(self):
        for bad in ("15/09/2024", "May 2025", "2025", "01-05-2025"):
            with self.assertRaises(ValidationError, msg=bad):
                DocumentMetadata(**{**VALID, "publication_date": bad})

    def test_impossible_date_is_rejected(self):
        """Passes the regex, is not a real day."""
        with self.assertRaises(ValidationError):
            DocumentMetadata(**{**VALID, "publication_date": "2024-13-45"})

    def test_document_type_is_a_closed_set(self):
        with self.assertRaises(ValidationError):
            DocumentMetadata(**{**VALID, "document_type": "technical report"})

    def test_keywords_must_be_a_list_of_at_least_two(self):
        with self.assertRaises(ValidationError):
            DocumentMetadata(**{**VALID, "keywords": ["only-one"]})
        with self.assertRaises(ValidationError):
            DocumentMetadata(**{**VALID, "keywords": "ai, ethics"})

    def test_keywords_are_lowercased_and_deduplicated(self):
        record = DocumentMetadata(**{**VALID, "keywords": ["AI", " ai ", "Ethics"]})
        self.assertEqual(record.keywords, ["ai", "ethics"])

    def test_placeholder_authors_become_none(self):
        """The model's many ways of saying "I don't know" collapse to one null."""
        for placeholder in ("Unknown", "N/A", "null", "not specified", "  ", "None"):
            record = DocumentMetadata(**{**VALID, "author": placeholder})
            self.assertIsNone(record.author, placeholder)

    def test_the_string_null_is_not_stored_as_a_date(self):
        record = DocumentMetadata(**{**VALID, "publication_date": "null"})
        self.assertIsNone(record.publication_date)

    def test_missing_author_and_date_are_allowed(self):
        record = DocumentMetadata(title="Untitled Report", keywords=["a", "b"],
                                  document_type="other")
        self.assertIsNone(record.author)
        self.assertIsNone(record.publication_date)


class TestGrounding(unittest.TestCase):
    """Schema checks shape; grounding checks truth."""

    def test_an_invented_date_is_detected(self):
        text = "This memo has no date anywhere in it."
        self.assertFalse(date_appears_in(text, "2023-04-01"))

    def test_a_reformatted_real_date_is_accepted(self):
        """The model is allowed to convert 15/09/2024 into 2024-09-15."""
        self.assertTrue(date_appears_in("Date of issue : 15/09/2024", "2024-09-15"))

    def test_written_month_names_are_accepted(self):
        self.assertTrue(date_appears_in("published 15 September 2024", "2024-09-15"))
        self.assertTrue(date_appears_in("on September 15, 2024", "2024-09-15"))

    def test_author_matched_on_a_name_token(self):
        text = "From: Priya Raman, Search Platform"
        self.assertTrue(author_appears_in(text, "Priya Raman, Search Platform"))
        self.assertFalse(author_appears_in(text, "Dr. Fake Person"))

    def test_verify_drops_an_unverified_date(self):
        record = DocumentMetadata(**{**VALID, "publication_date": "2023-04-01"})
        data, warnings = verify(record, "A document with no dates at all.")

        self.assertIsNone(data["publication_date"])
        self.assertTrue(any("does not appear" in w for w in warnings))

    def test_verify_keeps_a_date_that_is_present(self):
        # The source must mention the author too, or grounding correctly flags it —
        # an earlier version of this test omitted the name and failed for that
        # reason, which was the check working rather than a bug.
        source = "AI Trends 2025 by Dr. Sarah Lee, published 2025-05-01."
        record = DocumentMetadata(**{**VALID, "publication_date": "2025-05-01"})

        data, warnings = verify(record, source)

        self.assertEqual(data["publication_date"], "2025-05-01")
        self.assertEqual(data["author"], "Dr. Sarah Lee")
        self.assertEqual(warnings, [])

    def test_hints_refill_a_missing_author(self):
        data = {"author": None, "publication_date": None}
        data, recovered = recover_from_hints(data, {"author": "John Smith"})

        self.assertEqual(data["author"], "John Smith")
        self.assertTrue(recovered)

    def test_hints_do_not_overwrite_a_value_the_model_found(self):
        data = {"author": "Real Author", "publication_date": None}
        data, _ = recover_from_hints(data, {"author": "Meta Tag Author"})

        self.assertEqual(data["author"], "Real Author")


class TestLoaders(unittest.TestCase):
    def test_the_batch_loads_every_supported_format(self):
        documents = load_batch(DATA)
        formats = {d.fmt for d in documents}

        self.assertEqual(len(documents), 6)
        self.assertEqual(formats, {"txt", "md", "html", "docx"})

    def test_html_tags_are_stripped(self):
        document = load_html(DATA / "climate_agriculture.html")

        self.assertNotIn("<p>", document.text)
        self.assertNotIn("<html", document.text)
        self.assertIn("Shifting rainfall patterns", document.text)

    def test_html_head_metadata_is_kept_as_hints(self):
        document = load_html(DATA / "climate_agriculture.html")

        self.assertEqual(document.hints["author"], "John Smith")
        self.assertEqual(document.hints["date"], "2024-09-15")

    def test_docx_paragraphs_are_extracted(self):
        document = load_docx(DATA / "vendor_security_assessment.docx")

        self.assertIn("CloudVault", document.text)
        self.assertIn("Miguel Torres", document.text)
        self.assertNotIn("<w:t", document.text)

    def test_plain_text_is_read_unchanged(self):
        document = load_text(DATA / "ai_trends_2025.txt")
        self.assertIn("Dr. Sarah Lee", document.text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
