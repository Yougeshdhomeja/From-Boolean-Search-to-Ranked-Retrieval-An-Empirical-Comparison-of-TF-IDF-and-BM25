"""
Shared text preprocessing for the NLP poster project.

Provides one reusable preprocessing pipeline:
    lowercase -> alphabetic tokenization -> stopword removal -> Porter stemming

Used by document indexing and by every query-processing stage so that
all text is processed in exactly the same way.
"""

import re


# ---------------------------------------------------------------------------
# Stopword loading
# ---------------------------------------------------------------------------
def load_stopwords(stopwords_path):
    """Load stopwords, strip whitespace, ignore empty lines, return a set."""
    stop_words = set()
    with open(stopwords_path, encoding="utf-8") as f:
        for line in f:
            word = line.strip().lower()
            if word:
                stop_words.add(word)
    return stop_words


# ---------------------------------------------------------------------------
# Text preprocessing
# ---------------------------------------------------------------------------
def word_tokenization(text):
    """Lowercase the text and extract alphabetic word tokens."""
    text = text.lower()
    tokens = re.findall(r"[a-z]+", text)
    return tokens


def stemming_words(words, stemmer):
    """Apply Porter stemming to a list of words."""
    stemmed_words = []
    for word in words:
        stemmed_words.append(stemmer.stem(word))
    return stemmed_words


def preprocess_text(text, stop_words, stemmer):
    """
    Full preprocessing pipeline:
    1. lowercase + tokenize
    2. remove stopwords
    3. apply Porter stemming
    Returns the list of processed tokens for this document.
    """
    tokens = word_tokenization(text)

    filtered_tokens = []
    for token in tokens:
        if token not in stop_words:
            filtered_tokens.append(token)

    processed_tokens = stemming_words(filtered_tokens, stemmer)
    return processed_tokens
