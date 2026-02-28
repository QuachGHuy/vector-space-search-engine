import numpy as np
from collections import Counter
from scipy.sparse import lil_matrix, csr_matrix

from .tokenizer import tokenize

def create_doc_term_matrix_sparse(
        corpus: list[str], 
        vocabulary: list[str]
    ) -> csr_matrix:
    if not corpus or not vocabulary:
        raise ValueError("Corpus and vocabulary must not be empty")
    
    vocab_index = {w: i for i, w in enumerate(vocabulary)}
    matrix = lil_matrix((len(corpus), len(vocabulary)), dtype=np.int32)

    for doc_id, doc in enumerate(corpus):
        tokens = tokenize(doc)
        for token, count in Counter(tokens).items():
            if token in vocab_index:
                matrix[doc_id, vocab_index[token]] = count

    return csr_matrix(matrix)

if __name__ == "__main__":
    corpus = ["apple banana apple", "banana orange"]
    vocabulary = sorted(set(
        token for doc in corpus for token in tokenize(doc)
    ))

    m = create_doc_term_matrix_sparse(corpus, vocabulary)
    print(m.toarray())