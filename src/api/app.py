from fastapi import FastAPI
from src.api.lifespan import lifespan
from src.api.routers.search import router as search_router

def create_app() -> FastAPI:
    app = FastAPI(
        title="Vector Space Search Engine",
        lifespan=lifespan
    )

    app.include_router(search_router)

    return app