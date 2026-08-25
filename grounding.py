"""Check extracted values against the source text.

Pydantic validates SHAPE: is this a string, is the date YYYY-MM-DD, is
document_type one of the allowed values. It cannot validate TRUTH. A model asked
for the publication date of a document that has none returned "2023-04-01" — a
perfectly valid ISO date, accepted by the schema, and completely invented.

The only defence is to check the value against the document it supposedly came
from. That is a different kind of check and it belongs in a different place, which
is why it lives here rather than in a validator.

Scope, deliberately: this verifies the fields that can be checked mechanically —
a date and a person's name either appear in the text or they do not. Titles get a
softer check, and keywords are not checked at all, because a good keyword is
often a summary word that never appears verbatim.
"""

import re
from datetime import date

MONTHS = ("january", "february", "march", "april", "may", "june", "july",
          "august", "september", "october", "november", "december")


def date_appears_in(text: str, iso_date: str) -> bool:
    """True if the date is present in the text in any common format.

    The model is allowed to reformat "15/09/2024" into "2024-09-15", so a plain
    substring search is not enough. Every plausible written form of the same day
    is generated and searched for.
    """
    try:
        parsed = date.fromisoformat(iso_date)
    except ValueError:
        return False

    lowered = text.lower()
    day, month, year = parsed.day, parsed.month, parsed.year
    month_name = MONTHS[month - 1]

    candidates = {
        iso_date,
        f"{day:02d}/{month:02d}/{year}", f"{day}/{month}/{year}",
        f"{month:02d}/{day:02d}/{year}", f"{month}/{day}/{year}",
        f"{day:02d}-{month:02d}-{year}", f"{day}-{month}-{year}",
        f"{day:02d}.{month:02d}.{year}", f"{day}.{month}.{year}",
        f"{day} {month_name} {year}", f"{day:02d} {month_name} {year}",
        f"{month_name} {day}, {year}", f"{month_name} {day} {year}",
        f"{day} {month_name[:3]} {year}",
    }

    return any(candidate.lower() in lowered for candidate in candidates)


def author_appears_in(text: str, author: str) -> bool:
    """True if the author's name is in the text.

    Matched on surname rather than the whole string, because the model returns
    "Priya Raman, Search Platform" while the document says "Priya Raman" — an
    exact match would wrongly reject a correct answer. A name token of 3+
    characters appearing in the text is enough evidence that it was not invented.
    """
    lowered = text.lower()
    tokens = [t for t in re.split(r"[\s,.]+", author.lower()) if len(t) >= 3]

    if not tokens:
        return False

    return any(token in lowered for token in tokens)


def verify(metadata, source_text: str):
    """Compare metadata against its source. Returns (cleaned_dict, warnings).

    A field that cannot be found is set to None rather than kept. A missing value
    is honest; an invented one is a silent error that propagates into every
    downstream use of the dataset.
    """
    data = metadata.model_dump()
    warnings = []

    if data.get("publication_date"):
        if not date_appears_in(source_text, data["publication_date"]):
            warnings.append(
                f"publication_date {data['publication_date']!r} does not appear in the "
                f"source — dropped as unverified"
            )
            data["publication_date"] = None

    if data.get("author"):
        if not author_appears_in(source_text, data["author"]):
            warnings.append(
                f"author {data['author']!r} does not appear in the source — dropped as unverified"
            )
            data["author"] = None

    # Titles are checked loosely: the model may tidy capitalisation or drop a
    # subtitle, so a few overlapping words is the bar rather than an exact match.
    title_words = [w for w in re.split(r"\W+", data.get("title", "").lower()) if len(w) > 3]
    if title_words:
        lowered = source_text.lower()
        found = sum(1 for word in title_words if word in lowered)
        if found == 0:
            warnings.append(f"title {data['title']!r} shares no words with the source")

    return data, warnings


def recover_from_hints(data, hints):
    """Fill nulls from metadata the FILE FORMAT stated outright.

    HTML carries `<meta name="author">` and `<title>`. A value the format declares
    is better evidence than one a 3B model infers from prose, so it is used when
    the model returned nothing — this is what rescues the HTML document, where the
    model missed the byline it was told about twice.
    """
    recovered = []

    if not data.get("author") and hints.get("author"):
        data["author"] = hints["author"]
        recovered.append(f"author recovered from HTML meta tag: {hints['author']!r}")

    if not data.get("publication_date") and hints.get("date"):
        data["publication_date"] = hints["date"]
        recovered.append(f"publication_date recovered from HTML meta tag: {hints['date']!r}")

    return data, recovered
