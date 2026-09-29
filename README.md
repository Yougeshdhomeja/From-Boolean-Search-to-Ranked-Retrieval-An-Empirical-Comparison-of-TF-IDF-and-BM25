# From Boolean Search to Ranked Retrieval
## An Empirical Comparison of TF-IDF and BM25

This repository contains the implementation and reproducible analysis for an
NLP research poster comparing Boolean retrieval, TF-IDF cosine similarity,
and BM25 over a shared inverted index.

## Overview

The project investigates how three classical lexical retrieval methods behave
when they use the same corpus, preprocessing pipeline, and inverted index:

- Boolean retrieval
- TF-IDF cosine similarity
- BM25

The inverted index stores:

```text
term -> {document ID: term frequency}
```

and is shared by all three retrieval methods.

## Dataset

The experiment uses 50 documents.

Documents and queries are processed using:

1. lowercasing
2. alphabetic tokenization with `[a-z]+`
3. stopword removal
4. Porter stemming

## Retrieval Methods

### Boolean Retrieval

Supports:

- single-term queries
- AND queries using posting-list intersection
- OR queries using posting-list union

Boolean results are unranked.

### TF-IDF

TF-IDF uses raw term frequency and:

```text
IDF(t) = ln(N / DF(t))
```

Documents are ranked using cosine similarity with the full TF-IDF document
vector norm.

### BM25

The custom BM25 implementation uses:

```text
k1 = 1.5
b  = 0.75
```

with positive Lucene-style IDF:

```text
IDF(t) = ln(1 + (N - DF(t) + 0.5) / (DF(t) + 0.5))
```

## Reproducing the Experiment

Clone the repository and install the dependencies:

```bash
pip install -r requirements.txt
```

Then open:

```text
notebooks/poster_result_analysis.ipynb
```

and run all cells from top to bottom.

The notebook reproduces:

- dataset and inverted-index statistics
- representative retrieval queries
- the `passenger` TF-IDF/BM25 rank inversion
- exhaustive single-term top-rank agreement analysis

## Main RQ3 Result

Among comparable non-trivial single-term queries with document frequency
DF >= 2, TF-IDF and BM25 agree on the top-ranked document in:

```text
2,906 / 3,613 = 80.43%
```

Eight DF = 50 terms are excluded from this agreement denominator because
TF-IDF assigns them IDF = 0 and therefore returns no ranking.

## Project Structure

```text
Dataset/
    ShortStories/
    Stopword-List.txt

src/
    text_preprocessing.py
    inverted_index.py
    search_engine.py
    ranking_methods/
        tfidf.py
        bm25.py

notebooks/
    poster_result_analysis.ipynb

images/
    representative_queries_table.png
    passenger_rank_inversion.png
    top1_agreement_by_df.png
```

## Poster

**From Boolean Search to Ranked Retrieval — An Empirical Comparison of
TF-IDF and BM25**
