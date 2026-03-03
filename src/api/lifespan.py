from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI
import pandas as pd

from src.indexing import Indexer
from src.retrieval import SearchEngine

RAW_DATA_PATH = Path("data/raw/dataset.parquet")
INDEX_PATH = Path("data/index")

@asynccontextmanager
async def lifespan(app: FastAPI):
    index_files = [
        INDEX_PATH / "doc_term_matrix.npz",
        INDEX_PATH / "corpus.pkl",
        INDEX_PATH / "vocabulary.pkl"
    ]
    if not all(file.exists() for file in index_files): 
        dataset = pd.read_parquet(RAW_DATA_PATH)
        index = Indexer.build(dataset)
        Indexer.save(index, INDEX_PATH)
    
    index = Indexer.load(INDEX_PATH)

    app.state.search_engine = SearchEngine(
        corpus=index.corpus,
        vocabulary=index.vocabulary,
        doc_term_matrix=index.doc_term_matrix
    )

    yield