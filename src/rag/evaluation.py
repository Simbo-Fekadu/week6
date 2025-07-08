import pandas as pd
from .core import RAGSystem

def test_rag():
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
            "Sources": "\n- " + "\n- ".join(response["sources"]),
            "Relevant": keyword in response["answer"].lower()
        })
    
    pd.DataFrame(results).to_markdown("../reports/task3_test_results.md")

if __name__ == "__main__":
    test_rag()
