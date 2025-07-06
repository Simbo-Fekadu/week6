from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import pandas as pd
import chromadb

# Load cleaned data from Task 1
data = pd.read_csv('../data/filtered_complaints.csv')
texts = data['clean_text'].tolist()

# Split into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,  # Optimal for complaint narratives
    chunk_overlap=50  # Preserves context
)
chunks = splitter.create_documents(texts)

print(f"Generated {len(chunks)} chunks from {len(texts)} complaints")


# Initialize embedding model
embedder = SentenceTransformer('all-MiniLM-L6-v2')  # Lightweight but effective

# Embed chunks
chunk_texts = [chunk.page_content for chunk in chunks]
embeddings = embedder.encode(chunk_texts, show_progress_bar=True)

print(f"Embeddings shape: {embeddings.shape}")  # (num_chunks, 384)




client = chromadb.PersistentClient(path="../vector_store")
collection = client.create_collection(name="complaints")
collection.add(
    ids=[str(i) for i in range(len(chunks))],
    documents=chunk_texts,
    embeddings=embeddings.tolist()
)


# Quick test: Retrieve similar chunks
query = "unauthorized credit card charge"
query_embed = embedder.encode([query])
k = 3  # Top 3 matches
distances, indices = index.search(query_embed, k)

print("\nTop matches for query:")
for idx in indices[0]:
    print(f"- {chunks[idx].page_content[:200]}...")