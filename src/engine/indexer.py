import pickle
import pandas as pd
from pathlib import Path
from dataclasses import dataclass
from scipy.sparse import csr_matrix, save_npz, load_npz

from src.corpus import create_corpus
from src.vocabulary import create_vocabulary
from src.doc_term_matrix import create_doc_term_matrix_sparse

@dataclass
class Index:
    corpus: list[str]
    vocabulary: list[str]
    doc_term_matrix: csr_matrix

class Indexer:
    """
    Build, save and load vector-space search index.
    """
    @staticmethod
    def build(dataset: pd.DataFrame) -> Index:
        corpus = create_corpus(dataset)
        vocabulary = create_vocabulary(corpus)
        doc_term_matrix = create_doc_term_matrix_sparse(corpus, vocabulary)
        
        return Index(
            corpus=corpus, 
            vocabulary=vocabulary, 
            doc_term_matrix=doc_term_matrix
        )
    
    @staticmethod
    def save(index: Index, path: str | Path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        save_npz(path / "doc_term_matrix.npz", index.doc_term_matrix)

        with open(path / "corpus.pkl", "wb") as f:
            pickle.dump(index.corpus, f)

        with open(path / "vocabulary.pkl", "wb") as f:
            pickle.dump(index.vocabulary, f)

    @classmethod
    def load(cls, path: str | Path) -> Index:
        path = Path(path)

        if not (path / "doc_term_matrix.npz").exists():
            raise FileNotFoundError("Index files not found")
        
        doc_term_matrix = load_npz(path / "doc_term_matrix.npz")

        with open(path / "corpus.pkl", "rb") as f:
            corpus = pickle.load(f)

        with open(path / "vocabulary.pkl", "rb") as f:
            vocabulary = pickle.load(f)

        return Index(
            corpus=corpus,
            vocabulary=vocabulary,
            doc_term_matrix=doc_term_matrix
        )
    
if __name__ == "__main__":
    data = {
            "passages": [
                {
                    "passage_text": [
                        "machine learning is fun",
                        "deep learning is powerful"
                    ]
                },
                {
                    "passage_text": [
                        "banking and finance system",
                        "central bank regulation"
                    ]
                }
            ]
        }

    dataset = pd.DataFrame(data)

    # 1. Build index
    index = Indexer.build(dataset)

    print("Index built")
    print(f"- Corpus size: {len(index.corpus)}")
    print(f"- Vocabulary size: {len(index.vocabulary)}")
    print(f"- Doc-term matrix shape: {index.doc_term_matrix.shape}")

    test_path = Path("test_index")

    # 2. Save index
    Indexer.save(index, test_path)
    print("Index saved")

    # 3. Load index
    loaded_index = Indexer.load(test_path)
    print("Index loaded")

    # 4. Cleanup
    import shutil
    shutil.rmtree(test_path)


    
