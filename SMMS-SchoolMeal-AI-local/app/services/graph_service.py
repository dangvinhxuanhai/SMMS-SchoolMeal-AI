# app/services/graph_service.py
import os
from typing import List, Set, Dict
import pickle
import networkx as nx
from app.core.config import get_settings

settings = get_settings()

def _graph_path_for_school(school_id: str) -> str:
    # Trùng với build_graph.py
    return os.path.join(settings.DATA_DIR, f"ingredient_graph_{school_id}.gpickle")

class IngredientGraph:
    def __init__(self, graph: nx.Graph):
        self.graph = graph

    @classmethod
    def load_for_school(cls, school_id: str) -> "IngredientGraph":
        graph_path = _graph_path_for_school(school_id)

        if not os.path.exists(graph_path):
            raise RuntimeError(f"Graph file not found: {graph_path}")

        # Đọc graph bằng pickle
        with open(graph_path, "rb") as f:
            G = pickle.load(f)

        if not isinstance(G, nx.Graph):
            raise TypeError(f"Object loaded from {graph_path} is not a NetworkX graph")

        return cls(G)

    @classmethod
    def try_load_for_school(cls, school_id: str) -> "IngredientGraph | None":
        graph_path = _graph_path_for_school(school_id)
        if not os.path.exists(graph_path):
            return None

        try:
            return cls.load_for_school(school_id)
        except Exception:
            # TODO: log lỗi chi tiết
            return None
    # ----------------- NGUYÊN LIỆU (đã có) -----------------

    def ingredient_overlap_score(self, food_id: int, ingredient_ids: List[int]) -> float:
        food_node = f"Food:{food_id}"
        if food_node not in self.graph:
            return 0.0

        food_ing_ids: Set[int] = set()
        for n in self.graph.neighbors(food_node):
            if isinstance(n, str) and n.startswith("Ingredient:"):
                try:
                    iid = int(n.split(":")[1])
                    food_ing_ids.add(iid)
                except ValueError:
                    continue

        if not ingredient_ids:
            return 0.0

        overlap = len(food_ing_ids.intersection(ingredient_ids))
        return overlap / max(len(ingredient_ids), 1)

    # ----------------- DỊ ỨNG (mới) -----------------

    def get_food_allergen_ids(self, food_id: int) -> Set[int]:
        """
        Lấy tất cả AllergenId mà món này có thể liên quan,
        dựa trên đường Food -> Ingredient -> Allergen.
        """
        food_node = f"Food:{food_id}"
        if food_node not in self.graph:
            return set()

        allergen_ids: Set[int] = set()

        for ing_node in self.graph.neighbors(food_node):
            if not (isinstance(ing_node, str) and ing_node.startswith("Ingredient:")):
                continue

            # đi tiếp 1 bước: Ingredient -> Allergen
            for all_node in self.graph.neighbors(ing_node):
                if isinstance(all_node, str) and all_node.startswith("Allergen:"):
                    try:
                        aid = int(all_node.split(":")[1])
                        allergen_ids.add(aid)
                    except ValueError:
                        continue

        return allergen_ids

    def allergen_conflict_penalty(
        self,
        food_id: int,
        prevalence: Dict[int, float],  # {AllergenId: tỷ lệ HS bị dị ứng}
    ) -> float:
        """
        Tính penalty ∈ [0,1] cho món dựa trên tỷ lệ HS dị ứng từng allergen.

        - 0.0  = món không chứa allergen nào trong prevalence
        - gần 1.0 = món chứa phần lớn các allergen có tỷ lệ dị ứng cao
        """
        if not prevalence:
            return 0.0

        food_allergens = self.get_food_allergen_ids(food_id)
        if not food_allergens:
            return 0.0

        food_allergens = set(food_allergens)

        # tổng prevalence của tất cả allergen ta đang quan tâm (định chuẩn)
        denom = sum(prevalence.values())
        if denom <= 0:
            return 0.0

        # tổng prevalence của các allergen mà món này chứa
        num = sum(
            prevalence[aid]
            for aid in food_allergens
            if aid in prevalence
        )

        penalty = num / denom
        return float(max(0.0, min(1.0, penalty)))
