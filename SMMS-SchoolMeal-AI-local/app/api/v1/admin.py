# app/api/v1/admin.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.tasks.indexing import build_index_for_school
from app.tasks.graph_build import build_graph_for_school

router = APIRouter(prefix="/api/v1/admin", tags=["admin"])


class RebuildRequest(BaseModel):
    school_id: str
    rebuild_index: bool = True
    rebuild_graph: bool = True


class RebuildResponse(BaseModel):
    school_id: str
    indexed_food_count: int | None = None
    graph_nodes: int | None = None
    graph_edges: int | None = None


@router.post("/rebuild", response_model=RebuildResponse)
def rebuild_ai_assets(req: RebuildRequest):
    if not req.rebuild_index and not req.rebuild_graph:
        raise HTTPException(status_code=400, detail="Nothing to rebuild")

    indexed_count = None
    graph_nodes = None
    graph_edges = None

    if req.rebuild_index:
        indexed_count = build_index_for_school(req.school_id)

    if req.rebuild_graph:
        graph_nodes, graph_edges = build_graph_for_school(req.school_id)

    return RebuildResponse(
        school_id=req.school_id,
        indexed_food_count=indexed_count,
        graph_nodes=graph_nodes,
        graph_edges=graph_edges,
    )
