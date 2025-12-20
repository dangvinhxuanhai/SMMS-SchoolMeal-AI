from __future__ import annotations
from typing import List, Tuple, Dict
from uuid import UUID

from sqlalchemy.orm import Session

from app.models.schemas import RecommendRequest, RecommendResponse, DishDto
from app.services.rag_index import RagIndex, Candidate
from app.services.graph_service import IngredientGraph
from app.services.ml_ranker import MlRanker
from sqlalchemy import create_engine, text, bindparam
from app.core.config import get_settings

settings = get_settings()
_engine = create_engine(settings.SQLSERVER_CONN_STR)

def get_allergen_prevalence_for_school(
    school_id: str,
    allergen_ids: List[int],
) -> Dict[int, float]:
    """
    Trả về dict {AllergenId: tỷ lệ học sinh trong trường bị dị ứng} (0–1).
    Nếu không có học sinh hoặc allergen nào không có ai dính -> giá trị = 0.
    """
    if not allergen_ids:
        return {}

    # dùng expanding bindparam cho IN (...)
    sql = text(
        """
        WITH ActiveStudents AS (
            SELECT s.StudentId
            FROM school.Students s
            WHERE s.SchoolId = :school_id
              AND s.IsActive = 1
        ),
        Affected AS (
            SELECT sa.AllergenId,
                   COUNT(DISTINCT sa.StudentId) AS AffectedCount
            FROM nutrition.StudentAllergens sa
            JOIN ActiveStudents s
                 ON s.StudentId = sa.StudentId
            WHERE sa.AllergenId IN :allergen_ids
            GROUP BY sa.AllergenId
        )
        SELECT
            a.AllergenId,
            a.AffectedCount,
            (SELECT COUNT(*) FROM ActiveStudents) AS TotalStudents
        FROM Affected a;
        """
    ).bindparams(
        bindparam("allergen_ids", expanding=True)
    )

    with _engine.begin() as conn:
        rows = conn.execute(
            sql,
            {
                "school_id": school_id,
                # truyền list/tuple bình thường, SQLAlchemy sẽ expand thành IN (?, ?, ?)
                "allergen_ids": list(allergen_ids),
            },
        ).fetchall()

    prevalence: Dict[int, float] = {aid: 0.0 for aid in allergen_ids}
    total_students = 0

    for row in rows:
        total_students = row.TotalStudents or total_students
        if row.TotalStudents and row.TotalStudents > 0:
            prevalence[row.AllergenId] = row.AffectedCount / row.TotalStudents

    return prevalence

class MenuRecommender:
    """
    Recommender cho AI Menu.

    - ML Ranker: dùng chung cho mọi trường.
    - RagIndex & IngredientGraph: load theo SchoolId (per school) và cache trong bộ nhớ.
    """

    def __init__(self, ml_ranker: MlRanker):
        self.ml_ranker = ml_ranker
        # cache per-school để không phải load file liên tục
        self._rag_cache: Dict[str, RagIndex] = {}
        self._graph_cache: Dict[str, IngredientGraph] = {}

    @classmethod
    def load(cls) -> "MenuRecommender":
        """
        Khởi tạo recommender với ML ranker (nếu có model.pkl).
        RAG index + graph sẽ được load lazy theo từng school_id.
        """
        return cls(
            ml_ranker=MlRanker.load()
        )

    # ----------------- INTERNAL: LOAD INDEX / GRAPH THEO SCHOOL -----------------

    def _get_rag_for_school(self, school_id: UUID) -> RagIndex | None:
        key = str(school_id)
        if key not in self._rag_cache:
            rag = RagIndex.try_load_for_school(key)
            # cache luôn cả None để lần sau khỏi phải check file nữa
            self._rag_cache[key] = rag  # rag có thể là None
        return self._rag_cache[key]

    def _get_graph_for_school(self, school_id: UUID) -> IngredientGraph | None:
        key = str(school_id)
        if key not in self._graph_cache:
            graph = IngredientGraph.try_load_for_school(key)
            self._graph_cache[key] = graph
        return self._graph_cache[key]

    # ----------------- PUBLIC API -----------------

    def recommend(self, db: Session, req: RecommendRequest) -> RecommendResponse:
        # 0. Lấy index & graph cho đúng trường (có thể None nếu chưa build)
        print("=== [RECOMMEND] START ===")
        print("SchoolId:", req.school_id)
        print("UserId:", req.user_id)
        school_id = req.school_id.upper()
        rag_index = self._get_rag_for_school(school_id)
        graph = self._get_graph_for_school(school_id)

        print("RAG loaded:", rag_index is not None)
        print("Graph loaded:", graph is not None)

        # Nếu thiếu 1 trong 2 -> trả về rỗng + message, không crash
        if rag_index is None or graph is None:
            return RecommendResponse(
                recommended_main=[],
                recommended_side=[],
                message="AI menu chưa được build (thiếu file FAISS/graph) cho trường này. Vui lòng nhờ admin bấm nút Gen file AI."
            )

        # 1. build query text và retrieve candidates từ FAISS
        query_text = rag_index.build_query_text(
            req.main_ingredient_ids,
            req.side_ingredient_ids,
            req.avoid_allergen_ids,
            req.max_main_kcal,
            req.max_side_kcal
        )

        print("Query text:", query_text)

        candidates = rag_index.search_candidates(query_text, k=200)
        print("FAISS candidates count:", len(candidates))

        if candidates:
            c0 = candidates[0]
            print("Sample candidate:",
                c0.food_id, c0.food_name,
                "faiss=", c0.faiss_score,
                "kcal=", c0.total_kcal,
                "is_main=", c0.is_main_dish)

        # 2. tách main/side + filter kcal
        main_cands, side_cands = self._filter_candidates(req, candidates)

        print("Main candidates after filter:", len(main_cands))
        print("Side candidates after filter:", len(side_cands))
        # 3. tính score (FAISS + graph + ML)
        scored_main = self._score_candidates(req, graph, main_cands, is_main=True)
        scored_side = self._score_candidates(req, graph, side_cands, is_main=False)

        # 4. sort + cắt top K
        scored_main = sorted(scored_main, key=lambda x: x["final_score"], reverse=True)[:req.top_k_main]
        scored_side = sorted(scored_side, key=lambda x: x["final_score"], reverse=True)[:req.top_k_side]

        # 5. log session + results vào DB (rag.MenuRecommend*)
        #    ⚠ trước đây anh đang lỡ truyền query_text vào RequestJson, sửa lại thành req.json()
        request_json_str = req.json()
        # session_id = self._log_session_and_results(db, req.user_id, request_json_str, scored_main, scored_side)
        print("Top main returned:", len(scored_main))
        print("Top side returned:", len(scored_side))
        print("=== [RECOMMEND] END ===")
        # 6. build response
        return RecommendResponse(
            # session_id=session_id,
            recommended_main=[
                DishDto(
                    food_id=it["food_id"],
                    food_name=it["food_name"],
                    is_main_dish=True,
                    total_kcal=it["total_kcal"],
                    score=it["final_score"]
                )
                for it in scored_main
            ],
            recommended_side=[
                DishDto(
                    food_id=it["food_id"],
                    food_name=it["food_name"],
                    is_main_dish=False,
                    total_kcal=it["total_kcal"],
                    score=it["final_score"]
                )
                for it in scored_side
            ]
        )

    # ----------------- HELPERS -----------------

    def _filter_candidates(
        self,
        req: RecommendRequest,
        candidates: List[Candidate]
    ) -> Tuple[List[Candidate], List[Candidate]]:
        main_cands: List[Candidate] = []
        side_cands: List[Candidate] = []
        debug_count = 0
        for c in candidates:
            if c.is_main_dish:
                if req.max_main_kcal is not None and c.total_kcal is not None:
                    if c.total_kcal > req.max_main_kcal:
                        continue
                main_cands.append(c)
            else:
                if req.max_side_kcal is not None and c.total_kcal is not None:
                    if c.total_kcal > req.max_side_kcal:
                        continue
                side_cands.append(c)
            if debug_count < 3:
                print("---- Candidate ----")
                print("FoodId:", c.food_id, c.food_name)
                print("FAISS:", c.faiss_score)
                print("Total kcal:", c.total_kcal)
                debug_count += 1

        return main_cands, side_cands

    def _score_candidates(
        self,
        req: RecommendRequest,
        graph: IngredientGraph,
        candidates: List[Candidate],
        is_main: bool
    ) -> List[dict]:
        scored: List[dict] = []
        feature_list = []

        # danh sách allergen “có trong trường” mà handler đã truyền sang
        avoid_ids = req.avoid_allergen_ids or []

        # tính prevalence 1 lần cho trường này
        prevalence = get_allergen_prevalence_for_school(
            str(req.school_id),
            avoid_ids,
        )

        for c in candidates:
            # nguyên liệu dùng cho scoring
            ing_ids = req.main_ingredient_ids if is_main else req.side_ingredient_ids

            # score về nguyên liệu (graph)
            g_score = graph.ingredient_overlap_score(c.food_id, ing_ids)

            # penalty dị ứng dựa trên prevalence (0–1)
            allergen_penalty = graph.allergen_conflict_penalty(c.food_id, prevalence)

            # feature cho ML (giữ 3 feature cũ)
            feature_list.append({
                "faiss_score": c.faiss_score,
                "graph_score": g_score,
                "total_kcal": c.total_kcal or 0.0,
            })

            scored.append({
                "food_id": c.food_id,
                "food_name": c.food_name,
                "is_main_dish": is_main,
                "total_kcal": c.total_kcal,
                "faiss_score": c.faiss_score,
                "graph_score": g_score,
                "ml_score": 0.5,              # placeholder, update sau
                "allergen_penalty": allergen_penalty,
                # nếu bạn có sẵn allergen_ids trong Candidate thì giữ lại, không cũng được
                "final_score": 0.0,
            })
            if len(scored) < 3:
                print("Graph score:", g_score)
                print("Allergen penalty:", allergen_penalty)

        # ML scoring (nếu chưa có model thì trả 0.5 hết)
        ml_scores = self.ml_ranker.score_batch(feature_list)

        for i, ml_s in enumerate(ml_scores):
            scored[i]["ml_score"] = float(ml_s)

            base_score = (
                0.5 * scored[i]["faiss_score"] +
                0.3 * scored[i]["graph_score"] +
                0.2 * scored[i]["ml_score"]
            )

            penalty = scored[i]["allergen_penalty"]
            allergen_multiplier = max(0.0, 1.0 - 0.7 * penalty)

            scored[i]["final_score"] = base_score * allergen_multiplier

            if i < 3:
                print("---- SCORE BREAKDOWN ----")
                print("FoodId:", scored[i]["food_id"])
                print("FAISS:", scored[i]["faiss_score"])
                print("Graph:", scored[i]["graph_score"])
                print("ML:", scored[i]["ml_score"])
                print("Penalty:", penalty)
                print("Final:", scored[i]["final_score"])
        return scored


    def _log_session_and_results(
        self,
        db: Session,
        user_id: UUID,
        request_json: str,
        scored_main: List[dict],
        scored_side: List[dict]
    ) -> int:
        """
        Insert vào rag.MenuRecommendSessions & rag.MenuRecommendResults.
        Anh chỉnh lại tên schema/table nếu khác.

        Gợi ý: nếu sau này anh thêm cột SchoolId vào Sessions thì
        có thể truyền thêm req.school_id vào đây.
        """
        # INSERT session
        session_sql = """
        INSERT INTO rag.MenuRecommendSessions (UserId, CreatedAt, RequestJson, CandidateCount, ModelVersion)
        OUTPUT INSERTED.SessionId
        VALUES (:user_id, SYSDATETIME(), :req_json, :cand_count, :model_ver);
        """
        cand_count = len(scored_main) + len(scored_side)
        session_id = db.execute(
            text(session_sql),
            {
                "user_id": str(user_id),
                "req_json": request_json,
                "cand_count": cand_count,
                "model_ver": "v1.0"
            }
        ).scalar()

        # INSERT results
        result_sql = """
        INSERT INTO rag.MenuRecommendResults
            (SessionId, FoodId, IsMain, RankShown, Score, IsChosen, ChosenAt)
        VALUES
            (:session_id, :food_id, :is_main, :rank_shown, :score, 0, NULL);
        """

        rank = 1
        for it in scored_main:
            db.execute(
                text(result_sql),
                {
                    "session_id": session_id,
                    "food_id": it["food_id"],
                    "is_main": 1,
                    "rank_shown": rank,
                    "score": it["final_score"]
                }
            )
            rank += 1

        rank = 1
        for it in scored_side:
            db.execute(
                text(result_sql),
                {
                    "session_id": session_id,
                    "food_id": it["food_id"],
                    "is_main": 0,
                    "rank_shown": rank,
                    "score": it["final_score"]
                }
            )
            rank += 1

        db.commit()
        return session_id