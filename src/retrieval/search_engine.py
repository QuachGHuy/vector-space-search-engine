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
        self.doc_term_matrix = doc_term_matrix

        self.vocab_index = {w:i for i, w in enumerate(vocabulary)}
        self.vocab_size = len(vocabulary)

    def _vectorize_query(self, query: str) -> csr_matrix:
        return vectorize_sparse(query, self.vocab_index, self.vocab_size)
    
    def _score(self, query_vec: csr_matrix, doc_vec: csr_matrix) -> float:
        return cosine_similarity_sparse(query_vec, doc_vec)
    
    def search(self, query: str, top_k: int = 5) -> list[tuple[float, str]]:
        """
        Search top-k most relevant documents.
        Return: [(score, document_text)]
        """
        query_vec = self._vectorize_query(query)
        results = []

        if self.doc_term_matrix is None:
            raise ValueError("doc_term_matrix must not be None")
        
        for doc_id in range(self.doc_term_matrix.shape[0]):
            doc_vec = self.doc_term_matrix[doc_id]
            score = self._score(query_vec, doc_vec)

            if score > 0:
                results.append((score, self.corpus[doc_id]))
            
        results.sort(reverse=True, key=lambda x: x[0])
        
        return results[:top_k]
