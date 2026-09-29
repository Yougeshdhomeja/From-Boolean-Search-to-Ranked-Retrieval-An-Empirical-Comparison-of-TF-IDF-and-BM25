"""
NLP Document Search Engine for the Poster Project

Provides an interactive terminal menu for three retrieval methods:
    1. Boolean Retrieval
    2. TF-IDF Ranking
    3. BM25 Ranking

The corpus is loaded, preprocessed, and indexed exactly once at startup.
The resulting inverted index and document-length table are reused by all
three retrieval methods.
"""

from pathlib import Path
from nltk.stem import PorterStemmer

from text_preprocessing import (
    load_stopwords,
    preprocess_text,
)
from inverted_index import (
    load_documents,
    build_inverted_index,
)
from ranking_methods.tfidf import rank_tfidf
from ranking_methods.bm25 import rank_bm25


# ---------------------------------------------------------------------------
# Path setup relative to this script
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent

DATASET_DIR = REPO_ROOT / "Dataset"
DOCS_DIR = DATASET_DIR / "ShortStories"
STOPWORDS_PATH = DATASET_DIR / "Stopword-List.txt"


# ---------------------------------------------------------------------------
# Boolean query parsing
# ---------------------------------------------------------------------------
def parse_boolean_query(query):
    """
    Parse a simple Boolean query into (operator, terms).

    Supported query shapes:
        - single term:              "beard"
        - AND-only expression:      "god AND man AND love"
        - OR-only expression:       "horse OR donkey OR mule"

    Returns a tuple (operator, terms) where operator is one of:
        None, "AND", "OR", or "MIXED".

    Mixed AND/OR is detected and rejected with a clear message.
    Malformed operator placement or repeated operators are detected later.
    """
    # Normalize whitespace first
    query = query.strip()

    # Decide which operator is used by a simple count
    upper_query = query.upper()
    has_and = " AND " in upper_query or upper_query.startswith("AND ") or upper_query.endswith(" AND")
    has_or = " OR " in upper_query or upper_query.startswith("OR ") or upper_query.endswith(" OR")

    # Reject mixed AND/OR queries
    if has_and and has_or:
        return "MIXED", []

    # Choose the operator and split the query
    if has_and:
        operator = "AND"
        parts = query.split()
        # Filter out the operator tokens (case-insensitive)
        terms = []
        for part in parts:
            if part.upper() != "AND":
                terms.append(part)
    elif has_or:
        operator = "OR"
        parts = query.split()
        terms = []
        for part in parts:
            if part.upper() != "OR":
                terms.append(part)
    else:
        # Single-term query
        operator = None
        terms = [query]

    return operator, terms


def is_valid_query(operator, raw_terms, original_query):
    """
    Simple sanity check after parsing.
    Rejects queries that are empty, contain only operators,
    have no usable terms left, or have malformed operator placement.
    """
    stripped = original_query.strip()

    # Empty input or whitespace-only input
    if not stripped:
        return False

    # A single operator keyword as the entire query
    if operator is None and stripped.upper() in ("AND", "OR"):
        return False

    # No terms after filtering operators (e.g. "AND", "OR", "AND AND")
    if operator in ("AND", "OR") and not raw_terms:
        return False

    # Query starts or ends with an operator (e.g. "beard AND", "AND beard")
    upper = stripped.upper()
    if upper.startswith("AND ") or upper.startswith("OR "):
        return False
    if upper.endswith(" AND") or upper.endswith(" OR"):
        return False

    # Repeated operator tokens (e.g. "beard AND AND man", "beard OR OR man")
    if operator in ("AND", "OR"):
        op_upper = operator.upper()
        parts = stripped.split()
        prev = None
        for part in parts:
            current = part.upper()
            if current == op_upper and prev == op_upper:
                return False
            prev = current

    return True


# ---------------------------------------------------------------------------
# Query preprocessing
# ---------------------------------------------------------------------------
def preprocess_query_term(raw_term, stop_words, stemmer):
    """
    Preprocess a single raw query term using the same rules as documents.

    Returns:
        - processed token string if preprocessing yields exactly 1 token
        - None if preprocessing yields 0 tokens (missing/stopword operand)
        - raises ValueError if preprocessing yields more than 1 token
    """
    # Run the term through the same shared pipeline, but input is a single token
    processed_tokens = preprocess_text(raw_term, stop_words, stemmer)

    if len(processed_tokens) == 0:
        return None

    if len(processed_tokens) == 1:
        return processed_tokens[0]

    # More than one token means the operand was not a simple term
    raise ValueError(f"Query term {raw_term!r} produced multiple tokens: {processed_tokens}")


def preprocess_boolean_terms(raw_terms, stop_words, stemmer):
    """
    Preprocess each raw Boolean operand.

    Returns:
        - list of processed terms (None means the operand disappeared)
        - "INVALID_OPERAND" if any raw term produces more than one token
    """
    processed_terms = []
    for raw_term in raw_terms:
        try:
            processed = preprocess_query_term(raw_term, stop_words, stemmer)
        except ValueError:
            return "INVALID_OPERAND"
        processed_terms.append(processed)
    return processed_terms


# ---------------------------------------------------------------------------
# Postings retrieval and Boolean search
# ---------------------------------------------------------------------------
def get_postings(inverted_index, term):
    """Return the set of document IDs in which the term appears."""
    if term in inverted_index:
        return set(inverted_index[term].keys())
    return set()


def boolean_search(inverted_index, query, stop_words, stemmer):
    """
    Execute a simple Boolean search.

    Returns:
        - a sorted list of matching document IDs, or
        - "MIXED_OPERATORS" if the query mixes AND and OR.
    """
    operator, raw_terms = parse_boolean_query(query)

    if operator == "MIXED":
        return "MIXED_OPERATORS"

    if not is_valid_query(operator, raw_terms, query):
        return "INVALID_QUERY"

    # Preprocess each raw term individually
    processed_terms = preprocess_boolean_terms(raw_terms, stop_words, stemmer)

    if processed_terms == "INVALID_OPERAND":
        return "INVALID_QUERY"

    # Collect postings for each processed term.
    # A None processed term contributes an empty posting set.
    posting_sets = []
    for term in processed_terms:
        if term is None:
            posting_sets.append(set())
        else:
            posting_sets.append(get_postings(inverted_index, term))

    if operator is None:
        # Single-term search
        result = posting_sets[0]
    elif operator == "AND":
        # Intersection across all posting sets
        result = posting_sets[0]
        for postings in posting_sets[1:]:
            result = result.intersection(postings)
    elif operator == "OR":
        # Union across all posting sets
        result = set()
        for postings in posting_sets:
            result = result.union(postings)

    return sorted(result)


def format_boolean_result(result):
    """Format the search result for display."""
    if result == "MIXED_OPERATORS":
        return "Mixed AND/OR queries are not supported yet."
    if result == "INVALID_QUERY":
        return "Invalid query. Please enter a term, an AND query, or an OR query."
    if not result:
        return "No matching documents."
    return f"Matching documents:\n{result}"


# ---------------------------------------------------------------------------
# Ranked result formatting
# ---------------------------------------------------------------------------
def format_ranked_result(title, ranked_docs):
    """
    Format a list of (docID, score) tuples for display.

    Returns a human-readable string with numbered entries.
    """
    if not ranked_docs:
        return "No matching documents."

    lines = [f"{title}"]
    for rank, (doc_id, score) in enumerate(ranked_docs, start=1):
        lines.append(f"  {rank}. Document {doc_id:<5} Score: {score:.6f}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Menu and mode handlers
# ---------------------------------------------------------------------------
def print_menu():
    """Print the main retrieval-method menu."""
    print()
    print("=" * 50)
    print("NLP Document Search Engine")
    print("=" * 50)
    print("1. Boolean Retrieval")
    print("2. TF-IDF Ranking")
    print("3. BM25 Ranking")
    print("4. Exit")
    print("=" * 50)


def boolean_mode(inverted_index, stop_words, stemmer):
    """Run one Boolean query and display the result."""
    query = input("\nEnter Boolean query: ").strip()
    result = boolean_search(inverted_index, query, stop_words, stemmer)
    print(format_boolean_result(result))


def tfidf_mode(inverted_index, document_lengths, stop_words, stemmer):
    """Run one TF-IDF free-text ranking query and display results."""
    query = input("\nEnter TF-IDF query: ").strip()
    processed_query = preprocess_text(query, stop_words, stemmer)
    total_docs = len(document_lengths)
    ranked = rank_tfidf(inverted_index, processed_query, total_docs, top_k=10)
    print(format_ranked_result("TF-IDF Ranked Documents:", ranked))


def bm25_mode(inverted_index, document_lengths, stop_words, stemmer):
    """Run one BM25 free-text ranking query and display results."""
    query = input("\nEnter BM25 query: ").strip()
    processed_query = preprocess_text(query, stop_words, stemmer)
    total_docs = len(document_lengths)
    ranked = rank_bm25(inverted_index, document_lengths, processed_query, total_docs, top_k=10)
    print(format_ranked_result("BM25 Ranked Documents:", ranked))


def interactive_search(inverted_index, document_lengths, stop_words, stemmer):
    """Run the interactive menu loop."""
    while True:
        print_menu()
        choice = input("Choose method: ").strip()

        if choice == "1":
            boolean_mode(inverted_index, stop_words, stemmer)
        elif choice == "2":
            tfidf_mode(inverted_index, document_lengths, stop_words, stemmer)
        elif choice == "3":
            bm25_mode(inverted_index, document_lengths, stop_words, stemmer)
        elif choice == "4" or choice.lower() == "exit":
            print("Goodbye!")
            break
        else:
            print("Invalid choice. Please enter 1, 2, 3, or 4.")


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

    # Build the inverted index and document-length table once
    inverted_index, document_lengths = build_inverted_index(processed_documents)

    # Start the interactive search menu
    interactive_search(inverted_index, document_lengths, stop_words, stemmer)


if __name__ == "__main__":
    main()
