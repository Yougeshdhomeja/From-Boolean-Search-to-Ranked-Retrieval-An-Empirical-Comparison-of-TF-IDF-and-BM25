"""
TF-IDF Document Ranking for the NLP Poster Project

Ranks documents using classical TF-IDF weights and cosine similarity.

This module expects an already-built inverted index of the form:

    inverted_index[term][docID] = term_frequency

and a list of already-preprocessed query terms.  It does not load
Dataset files, perform text preprocessing, or build the index itself.
"""

import math


# ---------------------------------------------------------------------------
# IDF calculation
# ---------------------------------------------------------------------------
def calculate_idf_values(inverted_index, total_docs):
    """
    Calculate IDF for every indexed term.

    IDF(t) = ln(total_docs / DF(t))

    where DF(t) = len(inverted_index[t]) is the document frequency.

    Returns a dict {term: idf_value}.
    """
    idf_values = {}
    for term, postings in inverted_index.items():
        document_frequency = len(postings)
        # Every indexed term appears in at least one document
        idf_values[term] = math.log(total_docs / document_frequency)
    return idf_values


# ---------------------------------------------------------------------------
# Document vector norms
# ---------------------------------------------------------------------------
def calculate_document_norms(inverted_index, idf_values):
    """
    Calculate the full TF-IDF Euclidean norm for each document.

    norm(d) = sqrt(sum over all terms in d of (tf(t,d) * idf(t))^2)

    Returns a dict {docID: norm}.
    """
    squared_norms = {}

    for term, postings in inverted_index.items():
        idf = idf_values[term]
        for doc_id, term_freq in postings.items():
            weight = term_freq * idf
            squared_weight = weight * weight

            if doc_id not in squared_norms:
                squared_norms[doc_id] = 0.0
            squared_norms[doc_id] += squared_weight

    document_norms = {}
    for doc_id, squared_norm in squared_norms.items():
        document_norms[doc_id] = math.sqrt(squared_norm)

    return document_norms


# ---------------------------------------------------------------------------
# Query vector
# ---------------------------------------------------------------------------
def calculate_query_weights(query_terms, idf_values):
    """
    Build a TF-IDF query vector.

    query_tf(term)  = raw count of term in the query list
    query_weight(t) = query_tf(t) * idf(t)

    Returns a dict {term: query_weight} containing only known terms
    (terms that exist in idf_values).  Unknown terms contribute nothing.
    """
    query_tf = {}
    for term in query_terms:
        if term not in query_tf:
            query_tf[term] = 0
        query_tf[term] += 1

    query_weights = {}
    for term, tf in query_tf.items():
        if term in idf_values:
            query_weights[term] = tf * idf_values[term]

    return query_weights


def calculate_query_norm(query_weights):
    """
    Euclidean norm of the query TF-IDF vector.

    Returns 0.0 if the query vector is empty or all-zero.
    """
    squared_sum = 0.0
    for weight in query_weights.values():
        squared_sum += weight * weight
    return math.sqrt(squared_sum)


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
# Ranking
# ---------------------------------------------------------------------------
def rank_tfidf(inverted_index, query_terms, total_docs, top_k=10):
    """
    Rank documents for the given preprocessed query terms using TF-IDF cosine.

    Returns a list of (docID, score) tuples sorted by:
        1. cosine score descending
        2. docID ascending for ties

    If no query terms are known, returns an empty list.
    If the query vector norm is zero, returns an empty list.
    """
    # IDF values for the whole corpus
    idf_values = calculate_idf_values(inverted_index, total_docs)

    # Full document norms across all indexed terms
    document_norms = calculate_document_norms(inverted_index, idf_values)

    # Query TF-IDF vector (only known terms have weights)
    query_weights = calculate_query_weights(query_terms, idf_values)
    query_norm = calculate_query_norm(query_weights)

    # No known query terms or zero query norm -> nothing to rank
    if not query_weights or query_norm == 0.0:
        return []

    # Candidate documents: union of postings for known query terms
    candidate_docs = get_candidate_documents(inverted_index, query_weights.keys())
    if not candidate_docs:
        return []

    # Score each candidate
    scores = []
    for doc_id in candidate_docs:
        doc_norm = document_norms.get(doc_id, 0.0)
        if doc_norm == 0.0:
            continue

        dot_product = 0.0
        for term, query_weight in query_weights.items():
            if doc_id in inverted_index.get(term, {}):
                doc_weight = inverted_index[term][doc_id] * idf_values[term]
                dot_product += query_weight * doc_weight

        cosine_score = dot_product / (query_norm * doc_norm)
        if cosine_score > 0.0:
            scores.append((doc_id, cosine_score))

    # Sort by score descending, then docID ascending
    scores.sort(key=lambda item: (-item[1], item[0]))

    # Return top_k results
    return scores[:top_k]
