# app/services/graph_service.py
import os
from typing import List, Set
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

    def allergen_conflict_penalty(self, food_id: int, avoid_allergen_ids: List[int]) -> float:
        """
        Tính penalty ∈ [0,1] cho món theo danh sách allergen cần tránh.

        - 0.0  = không dính allergen nào trong avoid_allergen_ids
        - 0.5  = dính 50% số allergen user muốn tránh
        - 1.0  = dính toàn bộ allergen trong avoid_allergen_ids
        """
        if not avoid_allergen_ids:
            return 0.0

        avoid_set = set(avoid_allergen_ids)
        food_allergens = self.get_food_allergen_ids(food_id)

        if not food_allergens:
            return 0.0

        hit_count = len(food_allergens.intersection(avoid_set))
        if hit_count == 0:
            return 0.0

        penalty = hit_count / len(avoid_set)
        return float(max(0.0, min(1.0, penalty)))
