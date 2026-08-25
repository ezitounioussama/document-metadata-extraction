"""The Pydantic schema — where validation actually happens.

Worth being blunt about a trap in the brief's script: `JsonOutputParser` does
**not** validate against the Pydantic model. It uses the model only to build the
format instructions, and `parse()` returns a plain dict. Verified directly:

    parser = JsonOutputParser(pydantic_object=M)
    parser.parse('{"title": 123, "keywords": "not-a-list"}')
    # -> {'title': 123, 'keywords': 'not-a-list'}     accepted, no error

Validation happens only when the dict is passed through the model:
`DocumentMetadata(**parsed)`. That call is what raises on a wrong type or a
malformed date, and it is the step the brief's example is missing.
"""

import re
from datetime import date
from typing import List, Literal, Optional

from pydantic import BaseModel, Field, field_validator

# A closed set beats a free-text field: "report" / "Report" / "technical report"
# would otherwise all be distinct values in the output dataset, which breaks any
# downstream grouping.
DOCUMENT_TYPES = (
    "report",
    "article",
    "memo",
    "press_release",
    "assessment",
    "other",
)

ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


class DocumentMetadata(BaseModel):
    """The target schema. Every constraint here is one a real model output broke."""

    title: str = Field(
        ...,
        min_length=3,
        max_length=200,
        description="Title of the document",
    )

    # Optional, because a document genuinely may not name an author — the OCR scan
    # in this batch says "Author not recorded on the scan". Forcing a string would
    # push the model into inventing one, which is worse than a null.
    author: Optional[str] = Field(
        default=None,
        max_length=120,
        description="Author of the document, or null if the document does not name one",
    )

    publication_date: Optional[str] = Field(
        default=None,
        description="Publication date in YYYY-MM-DD format, or null if not stated",
    )

    keywords: List[str] = Field(
        ...,
        min_length=2,
        max_length=10,
        description="Between 2 and 10 topic keywords, lowercase",
    )

    document_type: Literal[DOCUMENT_TYPES] = Field(
        ...,
        description=f"One of: {', '.join(DOCUMENT_TYPES)}",
    )

    @field_validator("publication_date")
    @classmethod
    def check_iso_date(cls, value):
        """Reject anything that is not a real YYYY-MM-DD date.

        Two separate checks, because they catch different failures:

          * the regex rejects the wrong FORMAT — "15/09/2024" and "May 2025",
            both of which appear in this batch
          * date.fromisoformat rejects the right format with impossible VALUES —
            "2024-13-45" passes the regex and is not a date

        A string field with no validator would let both through and the "clean,
        structured dataset" would contain three different date formats.
        """
        # The model sometimes returns the STRING "null" rather than a JSON null,
        # which would otherwise be stored as a four-character date.
        if value is None:
            return None
        if str(value).strip().lower() in ("", "null", "none", "n/a", "not stated",
                                          "unknown", "not recorded"):
            return None

        if not ISO_DATE.match(value):
            raise ValueError(
                f"publication_date must be YYYY-MM-DD, got {value!r}"
            )

        try:
            date.fromisoformat(value)
        except ValueError as error:
            raise ValueError(f"publication_date {value!r} is not a real date") from error

        return value

    @field_validator("keywords")
    @classmethod
    def clean_keywords(cls, value):
        """Lowercase, strip, drop blanks and duplicates, keep order.

        Models return "Machine Learning", "machine learning" and " machine learning"
        interchangeably. Normalising here means the output dataset can be grouped
        and counted without a second cleaning pass.
        """
        seen = set()
        cleaned = []

        for keyword in value:
            item = str(keyword).strip().lower()
            if item and item not in seen:
                seen.add(item)
                cleaned.append(item)

        if len(cleaned) < 2:
            raise ValueError(f"need at least 2 distinct keywords, got {cleaned}")

        return cleaned

    @field_validator("author")
    @classmethod
    def blank_author_is_none(cls, value):
        """Turn the model's stand-ins for "unknown" into a real null.

        A 3B model asked for an author it cannot find returns "Unknown", "N/A",
        "Not specified" or an empty string. Left alone, those become four
        different values meaning the same thing.
        """
        if value is None:
            return None

        text = str(value).strip()
        if text.lower() in ("", "null", "unknown", "n/a", "na", "none",
                            "not specified", "not stated", "not recorded",
                            "author not recorded"):
            return None

        return text


class ExtractionResult(BaseModel):
    """One row of the output dataset: the metadata, or the reason there is none."""

    source_file: str
    format: str
    status: Literal["valid", "invalid"]
    metadata: Optional[DocumentMetadata] = None
    errors: List[str] = Field(default_factory=list)
    raw_model_output: Optional[str] = None
    attempts: int = 1
