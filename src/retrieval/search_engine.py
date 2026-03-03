import numpy as np
from scipy.sparse import csr_matrix

from src.preprocessing.vectorizer import vectorize_sparse
from .similarity import cosine_similarity_sparse

class SearchEngine:
    def __init__(
        self,
        corpus: list[str],
        vocabulary: list[str],
        doc_term_matrix: csr_matrix
    ):
        """
        Initialize search engine with pre-build index.
        """
        self.corpus = corpus
        self.vocabulary = vocabulary
        self.vocab_index = {w:i for i, w in enumerate(vocabulary)}
        self.vocab_size = len(vocabulary)

        self.doc_term_matrix = doc_term_matrix

    def _vectorize_query(self, query: str) -> csr_matrix:
        return vectorize_sparse(query, self.vocab_index, self.vocab_size)
    
    def _score(self, query_vec: csr_matrix, doc_vec: csr_matrix) -> np.ndarray:
        return cosine_similarity_sparse(query_vec, doc_vec)
    
    def search(self, query: str, top_k: int = 5) -> list[tuple[float, str]]:
        """
        Search top-k most relevant documents.
        Return: [(score, document_text)]
        """
        query_vec = self._vectorize_query(query)

        if self.doc_term_matrix is None:
            raise ValueError("doc_term_matrix must not be None")    

        scores = self._score(query_vec, self.doc_term_matrix)
        
        k = min(top_k, len(scores))

        top_indices = np.argpartition(scores, - k)[- k:]
        top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]

        results = [
            (float(scores[i]), self.corpus[i]) 
            for i in top_indices if scores[i] > 0
        ]
        return results
