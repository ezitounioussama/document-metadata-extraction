# Internal Memo: Search Relevance Regression

**From:** Priya Raman, Search Platform
**To:** Product leadership

We shipped the new ranking model on Tuesday and relevance dropped. This memo explains what
happened and what we are doing.

## What broke

The new model was trained on click data that had not been de-duplicated. Popular documents
appeared thousands of times, so the model learned popularity rather than relevance. Queries
with an obvious correct answer now return that answer on page two.

## What we are doing

Rolling back today. Re-training with de-duplicated data, and adding a held-out relevance set
to the deploy gate so this cannot ship silently again.

Keywords: search, ranking, relevance, regression, machine learning
