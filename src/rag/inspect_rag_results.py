# inspect_rag_results.py
from src.mcp.rag_tool import rag_search


def pretty_print_product(p):
    print("--------------------------------------------------")
    print(f"SKU:        {p.sku}")
    print(f"TITLE:      {p.title}")
    print(f"BRAND:      {p.brand}")
    print(f"PRICE:      {p.price}")
    # document_text may be long; just show a preview
    if getattr(p, "document_text", None):
        preview = p.document_text[:200].replace("\n", " ")
        print(f"TEXT PREVIEW: {preview}...")
    else:
        print("TEXT PREVIEW: <none>")
    print("RAW META KEYS:", list(p.raw_metadata.keys()))


if __name__ == "__main__":
    # Try a few queries that should match things in your catalog
    queries = [
        "window privacy film",
        "kids Frozen comforter",
        "hose clamps stainless steel",
    ]

    for q in queries:
        print("\n==============================")
        print("QUERY:", q)
        print("==============================")
        filters = {
            "$and": [
                {"main_category": {"$eq": "home & kitchen"}},
                {"price": {"$lte": 25.0}},
            ]
        }

        results = rag_search(q, top_k=10, filters=filters)
        if not results:
            print("No results.")
            continue

        for p in results:
            pretty_print_product(p)
