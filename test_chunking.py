
import json
from pathlib import Path

from src.document_loader import DocumentLoader
from src.text_splitter import RecursiveChunker


# STEP 1: Load original documents
loader = DocumentLoader("data/raw")
documents = loader.load_all_documents()

print("Documents loaded:", len(documents))


# STEP 2: Initialize chunker
chunker = RecursiveChunker(
    chunk_size=500,
    chunk_overlap=100
)


# STEP 3: Generate fresh chunks
chunks = chunker.chunk_documents(documents)

print("\nTotal chunks generated:", len(chunks))


# STEP 4: Display chunk information
for chunk in chunks:
    print("\n" + "=" * 70)
    print("CHUNK ID:", chunk["metadata"]["chunk_id"])
    print("SOURCE:", chunk["metadata"]["source"])
    print("CHUNK SIZE:", chunk["metadata"]["chunk_size"])
    print("CONTENT:", chunk["text"])


# STEP 5: Save regenerated chunks
output_path = Path("data/processed/chunks.json")

output_path.parent.mkdir(
    parents=True,
    exist_ok=True
)

with open(output_path, "w", encoding="utf-8") as file:
    json.dump(
        chunks,
        file,
        indent=4,
        ensure_ascii=False
    )

print("\nFresh chunks saved successfully!")
print("Location:", output_path)
