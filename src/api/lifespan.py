from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI
import pandas as pd

from src.indexing import Indexer
from src.retrieval import SearchEngine
from src.core.settings import settings

@asynccontextmanager
async def lifespan(app: FastAPI):
    index_files = [
        settings.INDEX_PATH / "doc_term_matrix.npz",
        settings.INDEX_PATH / "corpus.pkl",
        settings.INDEX_PATH / "vocabulary.pkl"
    ]
    if not all(file.exists() for file in index_files): 
        if not settings.RAW_DATA_PATH.exists():
            raise RuntimeError("RAW_DATA_PATH not found")
        
        dataset = pd.read_parquet(settings.RAW_DATA_PATH)
        index = Indexer.build(dataset)
        Indexer.save(index, settings.INDEX_PATH)
    
    index = Indexer.load(settings.INDEX_PATH)

    app.state.search_engine = SearchEngine(
        corpus=index.corpus,
        vocabulary=index.vocabulary,
        doc_term_matrix=index.doc_term_matrix,
    )

    yield