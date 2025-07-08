import os
import pandas as pd
from core import RAGSystem

# Set absolute output path
output_path = os.path.join(os.path.dirname(__file__), '..', 'reports', 'task3_results.csv')

def test_rag():
    # Create reports directory if missing
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    rag = RAGSystem()
    tests = [
        ("credit card", "unauthorized charges"),
        ("savings account", "overdraft fees")
    ]
    
    results = []
    for query, keyword in tests:
        response = rag.query(query)
        results.append({
            "Question": query,
            "Answer": response["answer"],
            "Sources": "\n".join(response["sources"][:2]),  # Show top 2 sources
            "Relevant": keyword in response["answer"].lower()
        })
    
    pd.DataFrame(results).to_csv(output_path, index=False)
    print(f"✅ Report saved to:\n{os.path.abspath(output_path)}")

if __name__ == "__main__":
    test_rag()
