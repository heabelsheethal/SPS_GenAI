"""
Implements word and sentence embeddings using spaCy,
following the Marimo notebook structure.
"""
# Get a word vector (nlp(word).vector)
# Compute cosine similarity between two words’ vectors
# Use Doc.vector for sentence embeddings (spaCy averages token vectors)


import spacy
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class WordEmbeddings:
    def __init__(self, model_name: str = "en_core_web_lg"):
        """
        Load the spaCy model (large model includes good word vectors).
        """
        try:
            self.nlp = spacy.load(model_name)
        except OSError as e:
            raise RuntimeError(
                f"spaCy model '{model_name}' not installed. Install it with:\n"
                f"    python -m spacy download {model_name}\n"
                f"Error: {e}"
            )

    def calculate_embedding(self, word: str) -> list[float]:
        """
        Return embedding vector for a single word.
        Equivalent to notebook's calculate_embedding().
        """
        doc = self.nlp(word)
        if not doc[0].has_vector:
            raise ValueError(f"No embedding found for word '{word}'")
        return doc.vector.tolist()

    def calculate_similarity(self, word1: str, word2: str) -> float:
        """
        Compute cosine similarity between two words.
        Equivalent to notebook's calculate_similarity().
        """
        vec1 = self.nlp(word1).vector
        vec2 = self.nlp(word2).vector
        return float(cosine_similarity([vec1], [vec2])[0][0])

    def sentence_similarity(self, query: str, candidates: list[str]) -> dict[str, float]:
        """
        Compute similarity between a query sentence and multiple candidate sentences.
        Equivalent to the notebook's query/info examples.
        Returns a dict mapping candidate -> similarity score.
        """
        q_doc = self.nlp(query)
        results = {}
        for text in candidates:
            cand_doc = self.nlp(text)
            results[text] = float(q_doc.similarity(cand_doc))
        return results

    def get_sentence_embedding(self, sentence: str) -> list[float]:
        """
        Return the embedding vector for a sentence.
        (spaCy Doc.vector = mean of token vectors).
        """
        doc = self.nlp(sentence)
        return doc.vector.tolist()
