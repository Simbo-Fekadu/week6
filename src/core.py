from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer

class RAGSystem:
    def __init__(self):
        self.client = PersistentClient(path="../vector_store")
        self.collection = self.client.get_collection("complaints")
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')

    def query(self, question: str, top_k: int = 5) -> list:
        # Step 1: Retrieve relevant complaints
        embeddings = self.embedder.encode(question).tolist()
        results = self.collection.query(
            query_embeddings=[embeddings],
            n_results=top_k,
            include=["documents", "metadatas"]
        )
        
        # Step 2: Format response (mock LLM for now)
        sources = [doc[:200] + "..." for doc in results['documents'][0]]
        return {
            "answer": f"Mock: Users report issues with {question.split()[-1]} (Task 4 will add real LLM)",
            "sources": sources
        }
