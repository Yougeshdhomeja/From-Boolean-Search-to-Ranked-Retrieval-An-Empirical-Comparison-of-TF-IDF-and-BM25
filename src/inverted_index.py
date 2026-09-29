"""
Inverted Index Builder for NLP Poster Project

Builds a classical inverted index from the Dataset/ShortStories corpus.
Steps: load documents -> preprocess (shared pipeline) -> index.

The index stores term frequencies per document so that later stages
(Boolean retrieval, TF-IDF, BM25) can reuse the same statistics.

This file intentionally uses simple procedural Python for clarity.
"""

from pathlib import Path
from nltk.stem import PorterStemmer

from text_preprocessing import (
    load_stopwords,
    preprocess_text,
)


# ---------------------------------------------------------------------------
# Path setup relative to this script
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent

DATASET_DIR = REPO_ROOT / "Dataset"
DOCS_DIR = DATASET_DIR / "ShortStories"
STOPWORDS_PATH = DATASET_DIR / "Stopword-List.txt"


# ---------------------------------------------------------------------------
# Document loading
# ---------------------------------------------------------------------------
def load_documents(docs_dir):
    """
    Load all .txt documents from docs_dir.
    Document IDs are taken from the numeric filename (1.txt -> 1).
    Returns a list of (docID, text) tuples sorted by docID.
    """
    doc_files = sorted(docs_dir.glob("*.txt"), key=lambda p: int(p.stem))

    documents = []
    for doc_path in doc_files:
        doc_id = int(doc_path.stem)
        text = doc_path.read_text(encoding="utf-8")
        documents.append((doc_id, text))

    return documents


# ---------------------------------------------------------------------------
# Inverted index construction
# ---------------------------------------------------------------------------
def build_inverted_index(processed_docs):
    """
    Build the inverted index and document-length table.

    processed_docs: list of (docID, tokens) tuples.

    Returns:
        inverted_index: dict {term: {docID: term_frequency}}
        document_lengths: dict {docID: number_of_indexed_tokens}
    """
    inverted_index = {}
    document_lengths = {}

    for doc_id, tokens in processed_docs:
        # Record how many tokens this document contributes after preprocessing
        document_lengths[doc_id] = len(tokens)

        # Update posting list for each token
        for term in tokens:
            if term not in inverted_index:
                inverted_index[term] = {}

            if doc_id not in inverted_index[term]:
                inverted_index[term][doc_id] = 0

            inverted_index[term][doc_id] += 1

    return inverted_index, document_lengths


# ---------------------------------------------------------------------------
# Summary / sanity-check output
# ---------------------------------------------------------------------------
def print_summary(inverted_index, document_lengths):
    """Print a concise summary of the built index."""
    num_docs = len(document_lengths)
    vocab_size = len(inverted_index)

    print("=" * 50)
    print("Inverted Index Summary")
    print("=" * 50)
    print(f"Documents processed: {num_docs}")
    print(f"Vocabulary size:   {vocab_size}")
    print()

    # Document length sample
    print("Sample document lengths (indexed tokens):")
    for doc_id in [1, 2, 5, 10, 50]:
        if doc_id in document_lengths:
            print(f"  doc {doc_id}: {document_lengths[doc_id]} tokens")
    print()

    # Sample index entries
    sample_terms = ["beard", "passeng", "love", "man", "ball"]
    print("Sample index entries:")
    for term in sample_terms:
        if term in inverted_index:
            postings = inverted_index[term]
            print(f"  {term}: {len(postings)} docs -> {dict(postings)}")
    print()

    # Document-frequency / term-frequency sanity checks
    if "beard" in inverted_index:
        beard_postings = inverted_index["beard"]
        print(f"'beard' document frequency: {len(beard_postings)}")
        print(f"'beard' postings sorted:    {sorted(beard_postings.keys())}")
        print()

    # Find an example term with term_frequency > 1 in some document
    print("Term-frequency example (term appears more than once in a doc):")
    found = False
    for term, postings in inverted_index.items():
        for doc_id, freq in postings.items():
            if freq > 1:
                print(f"  '{term}' in doc {doc_id}: frequency = {freq}")
                found = True
                break
        if found:
            break
    print()

    total_indexed_tokens = sum(document_lengths.values())
    avg_doc_length = total_indexed_tokens / num_docs if num_docs else 0
    print(f"Total indexed tokens: {total_indexed_tokens}")
    print(f"Average doc length:   {avg_doc_length:.2f} tokens")
    print("=" * 50)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def main():
    # Load stopwords once
    stop_words = load_stopwords(STOPWORDS_PATH)

    # Initialize the Porter stemmer
    stemmer = PorterStemmer()

    # Load raw documents
    raw_documents = load_documents(DOCS_DIR)

    # Preprocess every document
    processed_documents = []
    for doc_id, text in raw_documents:
        tokens = preprocess_text(text, stop_words, stemmer)
        processed_documents.append((doc_id, tokens))

    # Build the inverted index and document-length table
    inverted_index, document_lengths = build_inverted_index(processed_documents)

    # Print a concise summary
    print_summary(inverted_index, document_lengths)


if __name__ == "__main__":
    main()
