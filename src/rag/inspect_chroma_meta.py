import chromadb
from chromadb.config import Settings

from src.config import CHROMA_COLLECTION_NAME, VECTOR_STORE_DIR


def main():
    client = chromadb.PersistentClient(
        path=str(VECTOR_STORE_DIR),
        settings=Settings(anonymized_telemetry=False),
    )
    col = client.get_collection(CHROMA_COLLECTION_NAME)

    peek = col.peek(5)  # look at 5 docs
    print("IDS:", peek.get("ids"))
    print("METADATAS:")
    for m in peek.get("metadatas", []):
        print(m)


if __name__ == "__main__":
    main()
