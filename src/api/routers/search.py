from fastapi import APIRouter, Request, Query, HTTPException
from .schemas import SearchResultItem, SearchResponse

router = APIRouter(prefix="/search", tags=["search"])

@router.get("", response_model=SearchResponse)
def search(
    request: Request,
    q: str = Query(..., min_length=1, max_length=256),
    top_k: int = Query(5, ge=1, le=50)
):
    engine = getattr(request.app.state, "search_engine", None)
    if engine is None:
        raise HTTPException(
            status_code=503,
            detail="Search engine not initialized"
        )
    
    try:
        results = engine.search(q, top_k)
    except Exception as e:
        raise HTTPException(500, str(e))

    return SearchResponse(
        query=q,
        top_k=top_k,
        results=[SearchResultItem(score=score, document=doc) for (score, doc) in results]
    )