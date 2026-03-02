import numpy as np

def cosine_similarity_sparse(a, b):
    # Return 0 if either vector has no non-zero elements
    if a.nnz == 0 or b.nnz == 0:
        return 0.0
    
    numerator = (a @ b.T)[0, 0]
    denominator = np.linalg.norm(a.data) * np.linalg.norm(b.data)

    return numerator / denominator

if __name__ == '__main__':
    from scipy.sparse import csr_matrix

    vec_a = csr_matrix([[1, 2, 1]])
    vec_b = csr_matrix([[1, 1, 1]])

    print(cosine_similarity_sparse(vec_a, vec_b))