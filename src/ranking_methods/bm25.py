"""
BM25 Document Ranking for the NLP Poster Project

Ranks documents using the probabilistic BM25 scoring model.

This module expects an already-built inverted index:

    inverted_index[term][docID] = term_frequency

and a document-length table:

    document_lengths[docID] = number_of_indexed_tokens

It does NOT load Dataset files, preprocess text, build the index, or
implement Boolean search.  Query terms must already be preprocessed.
"""

import math


# ---------------------------------------------------------------------------
# BM25 parameters
# ---------------------------------------------------------------------------
DEFAULT_K1 = 1.5
DEFAULT_B = 0.75


# ---------------------------------------------------------------------------
# IDF calculation
# ---------------------------------------------------------------------------
def calculate_bm25_idf_values(inverted_index, total_docs):
    """
    Calculate positive BM25 IDF for every indexed term.

    IDF(t) = ln(1 + (N - DF(t) + 0.5) / (DF(t) + 0.5))

    where DF(t) = len(inverted_index[t]) is the document frequency.

    Returns a dict {term: idf_value}.
    """
    idf_values = {}
    for term, postings in inverted_index.items():
        document_frequency = len(postings)
        numerator = total_docs - document_frequency + 0.5
        denominator = document_frequency + 0.5
        idf_values[term] = math.log(1.0 + numerator / denominator)
    return idf_values


# ---------------------------------------------------------------------------
# Average document length
# ---------------------------------------------------------------------------
def calculate_average_document_length(document_lengths):
    """
    Calculate the average number of indexed tokens per document.

    AVGDL = sum(document_lengths.values()) / len(document_lengths)
    """
    total_length = sum(document_lengths.values())
    num_docs = len(document_lengths)
    if num_docs == 0:
        return 0.0
    return total_length / num_docs


# ---------------------------------------------------------------------------
# Query term frequencies
# ---------------------------------------------------------------------------
def calculate_query_term_frequencies(query_terms, idf_values):
    """
    Count raw query term frequencies for known terms only.

    Unknown terms are ignored and do not create candidates.
    """
    query_tf = {}
    for term in query_terms:
        if term in idf_values:
            if term not in query_tf:
                query_tf[term] = 0
            query_tf[term] += 1
    return query_tf


# ---------------------------------------------------------------------------
# Candidate documents
# ---------------------------------------------------------------------------
def get_candidate_documents(inverted_index, query_terms):
    """
    Candidate documents are the union of posting lists for known query terms.

    Returns a set of doc IDs.
    """
    candidates = set()
    for term in query_terms:
        if term in inverted_index:
            candidates.update(inverted_index[term].keys())
    return candidates


# ---------------------------------------------------------------------------
# BM25 term scoring
# ---------------------------------------------------------------------------
def calculate_bm25_term_score(term, doc_id, inverted_index, idf_values,
                              document_lengths, avg_doc_length, k1, b):
    """
    Calculate the BM25 contribution of a single query term in one document.

    term_score =
        IDF(t)
        *
        (TF * (k1 + 1))
        /
        (TF + k1 * (1 - b + b * (DL / AVGDL)))
    """
    tf = inverted_index[term][doc_id]
    idf = idf_values[term]
    doc_length = document_lengths[doc_id]

    # Document-length normalization factor
    length_factor = 1.0 - b + b * (doc_length / avg_doc_length)

    numerator = tf * (k1 + 1.0)
    denominator = tf + k1 * length_factor

    return idf * (numerator / denominator)


# ---------------------------------------------------------------------------
# Ranking
# ---------------------------------------------------------------------------
def rank_bm25(inverted_index, document_lengths, query_terms, total_docs,
              k1=DEFAULT_K1, b=DEFAULT_B, top_k=10):
    """
    Rank documents for the given preprocessed query terms using BM25.

    The final BM25 score for a document is the sum of each known query term's
    BM25 contribution, multiplied by the raw query frequency of that term.

    Returns a list of (docID, score) tuples sorted by:
        1. BM25 score descending
        2. docID ascending for ties

    If no known query terms exist, returns an empty list.
    If the total score for a candidate is zero or negative, it is excluded.
    """
    # BM25 IDF values for the whole corpus
    idf_values = calculate_bm25_idf_values(inverted_index, total_docs)

    # Average document length across the corpus
    avg_doc_length = calculate_average_document_length(document_lengths)

    # Query term frequencies (only known terms)
    query_tf = calculate_query_term_frequencies(query_terms, idf_values)
    if not query_tf:
        return []

    # Candidate documents: union of postings for known query terms
    candidate_docs = get_candidate_documents(inverted_index, query_tf.keys())
    if not candidate_docs:
        return []

    # Score each candidate
    scores = []
    for doc_id in candidate_docs:
        doc_score = 0.0
        for term, query_freq in query_tf.items():
            if doc_id in inverted_index.get(term, {}):
                term_score = calculate_bm25_term_score(
                    term, doc_id, inverted_index, idf_values,
                    document_lengths, avg_doc_length, k1, b
                )
                doc_score += query_freq * term_score

        if doc_score > 0.0:
            scores.append((doc_id, doc_score))

    # Sort by score descending, then docID ascending
    scores.sort(key=lambda item: (-item[1], item[0]))

    # Return top_k results
    return scores[:top_k]
