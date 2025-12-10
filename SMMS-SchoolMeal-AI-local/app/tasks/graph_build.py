# app/tasks/graph_build.py
import os
import pickle
import networkx as nx
from sqlalchemy import create_engine, text
from app.core.config import get_settings

settings = get_settings()


def _get_engine():
    return create_engine(settings.SQLSERVER_CONN_STR)


def build_graph_for_school(school_id: str) -> tuple[int, int]:
    """
    Build graph Ingredient + Allergen cho 1 trường.
    Trả về (số node, số edge).
    """
    engine = _get_engine()

    sql = """
    SELECT
        f.FoodId,
        fi.IngredientId,
        ai.AllergenId
    FROM nutrition.FoodItems f
    JOIN nutrition.FoodItemIngredients fi
         ON fi.FoodId = f.FoodId
    JOIN nutrition.Ingredients ing
         ON ing.IngredientId = fi.IngredientId
    LEFT JOIN nutrition.AllergeticIngredients ai
         ON ai.IngredientId = ing.IngredientId
    WHERE f.IsActive = 1
      AND f.SchoolId = :school_id;
    """

    with engine.connect() as conn:
        rows = conn.execute(text(sql), {"school_id": school_id}).fetchall()

    G = nx.Graph()

    if rows:
        for food_id, ing_id, allergen_id in rows:
            f_node = f"Food:{food_id}"
            i_node = f"Ingredient:{ing_id}"

            G.add_node(f_node)
            G.add_node(i_node)
            G.add_edge(f_node, i_node, kind="has_ingredient")

            if allergen_id is not None:
                a_node = f"Allergen:{allergen_id}"
                G.add_node(a_node)
                G.add_edge(i_node, a_node, kind="has_allergen")

    os.makedirs(settings.DATA_DIR, exist_ok=True)
    graph_path = os.path.join(settings.DATA_DIR, f"ingredient_graph_{school_id}.gpickle")

    with open(graph_path, "wb") as f:
        pickle.dump(G, f, protocol=pickle.HIGHEST_PROTOCOL)

    return G.number_of_nodes(), G.number_of_edges()
