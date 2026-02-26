from scipy.sparse import csr_matrix

from .vectorizer import vectorize_sparse
from .similarity import cosine_similarity_sparse

def ranking(query: str, vocab_index: dict[str, int], doc_term_matrix: csr_matrix) -> list[tuple[int, float]]:
    if doc_term_matrix is None:
        raise ValueError("doc_term_matrix must be created before ranking")
    
    vocab_size = len(vocab_index)
    query_vec = vectorize_sparse(query, vocab_index, vocab_size)

    scores: list[tuple[int, float]] = []

    for doc_id in range(doc_term_matrix.shape[0]):
        doc_vec = doc_term_matrix[doc_id]
        similarity = cosine_similarity_sparse(query_vec, doc_vec)
        scores.append((doc_id, similarity))

    scores.sort(reverse=True, key=lambda x: x[1])
    
    return scores

