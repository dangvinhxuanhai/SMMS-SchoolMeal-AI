# scripts/build_graph.py
import os
import pickle
import networkx as nx
from sqlalchemy import create_engine, text

from app.core.config import get_settings

settings = get_settings()


def main():
    os.makedirs(settings.DATA_DIR, exist_ok=True)
    engine = create_engine(settings.SQLSERVER_CONN_STR)

    school_id = os.getenv("SCHOOL_ID")
    if not school_id:
        raise RuntimeError("Vui lòng set biến môi trường SCHOOL_ID (GUID của trường)")

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

    from sqlalchemy import text as sql_text
    with engine.connect() as conn:
        rows = conn.execute(sql_text(sql), {"school_id": school_id}).fetchall()

    G = nx.Graph()

    for food_id, ing_id, allergen_id in rows:
        f_node = f"Food:{food_id}"
        i_node = f"Ingredient:{ing_id}"

        # Food - Ingredient
        G.add_node(f_node)
        G.add_node(i_node)
        G.add_edge(f_node, i_node, kind="has_ingredient")

        # Ingredient - Allergen (nếu có)
        if allergen_id is not None:
            a_node = f"Allergen:{allergen_id}"
            G.add_node(a_node)
            G.add_edge(i_node, a_node, kind="has_allergen")

    graph_path = os.path.join(settings.DATA_DIR, f"ingredient_graph_{school_id}.gpickle")

    # ✅ NetworkX 3.x: dùng pickle thay cho nx.write_gpickle
    with open(graph_path, "wb") as f:
        pickle.dump(G, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"✅ Graph saved: {graph_path}")
    print(f"   Nodes: {G.number_of_nodes()}, edges: {G.number_of_edges()}")


if __name__ == "__main__":
    main()
