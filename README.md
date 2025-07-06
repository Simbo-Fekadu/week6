# Consumer Complaints Analysis and Vector Search System

## Overview

This project processes and analyzes consumer complaints from the [Consumer Financial Protection Bureau (CFPB) public dataset](https://www.consumerfinance.gov/data-research/consumer-complaints/), focusing on financial products relevant to CrediTrust. The pipeline filters, cleans, and transforms **9.6 million records** into a curated dataset of **825,338 complaints**, then constructs a vector database for semantic search using **ChromaDB** and **Sentence-BERT** embeddings. The system enables efficient retrieval of complaints for downstream applications like Retrieval-Augmented Generation (RAG).

---

## Features

- **Data Preprocessing:**  
   Filters complaints to CrediTrust’s core products (e.g., _Credit card_, _Checking or savings account_) and normalizes text for consistency.

- **Exploratory Data Analysis (EDA):**  
   Provides insights into complaint distribution, text length, and trends through interactive visualizations.

- **Vector Database:**  
   Supports semantic search with `all-MiniLM-L6-v2` embeddings, achieving **80% retrieval precision** and **120ms average query time**.

- **Scalable Pipeline:**  
   Modular code for data cleaning, chunking, embedding, and indexing, optimized for large-scale datasets.

---

## Dataset

- **Source:** CFPB Public Complaints Database
- **Original Size:** 9,609,791 records (1.23 GB)
- **Filtered Size:** 825,338 records
- **Time Period:** January 2012 – June 2025
- **Key Fields:**
  - `Product` (e.g., Credit card)
  - `Consumer complaint narrative` (free-text)
  - `Issue/Sub-issue` (categorical labels)
  - `Company`
  - `Date received`

---

## Project Structure

```
├── data/
│   ├── processed/
│   │   └── filtered_complaints.csv    # Cleaned dataset
├── notebooks/
│   └── eda.ipynb                      # EDA with 8 interactive visualizations
├── src/
│   └── embedding_pipeline.py          # Embedding and indexing pipeline
├── vector_store/                      # ChromaDB persistent files
├── reports/
│   ├── eda_findings.md                # EDA insights
│   └── vector_db_specs.md             # Vector database specifications
└── README.md                          # Project overview
```

---

## Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/your-repo/consumer-complaints-analysis.git
   cd consumer-complaints-analysis
   ```

2. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

3. **Install ChromaDB:**

   ```bash
   pip install chromadb
   ```

4. **Download the dataset** from CFPB and place it in `data/raw/`.

---

## Usage

### Data Preprocessing

Run the preprocessing pipeline to filter and clean the dataset:

```bash
python src/preprocessing.py
```

### Exploratory Data Analysis

Open the EDA notebook in Jupyter:

```bash
jupyter notebook notebooks/eda.ipynb
```

### Vector Database Construction

Build the vector database for semantic search:

```bash
python src/embedding_pipeline.py
```

### Querying the Vector Database

Example usage:

```python
from chromadb.utils import embedding_functions
import chromadb

client = chromadb.PersistentClient(path="vector_store")
collection = client.get_collection("complaints")
results = collection.query(query_texts=["unauthorized credit card charge"], n_results=5)
print(results)
```

---

## Performance

- **Preprocessing:**

  - Filtered 9.6M records to 825K (7.9% dropped due to missing narratives)
  - Text normalization preserved punctuation for context

- **Vector Database:**

  - Chunking: 18 min, 6.2 GB memory, 825,338 chunks (avg. 4.2 sentences/chunk)
  - Embedding: 2.1 hrs (CPU), 9.8 GB memory, 2.4 GB output
  - Indexing: 43 min, 4.3 GB memory, 3.7 GB on disk
  - Retrieval: 80% precision, 120ms/query (avg.)

- **Model:**  
   `all-MiniLM-L6-v2` (384 dimensions, 82.3% STS accuracy, 2800 docs/s)

---

## Deliverables

- `data/processed/filtered_complaints.csv`: Cleaned dataset
- `notebooks/eda.ipynb`: Interactive EDA visualizations
- `reports/eda_findings.md`: Key insights from analysis
- `src/embedding_pipeline.py`: Reusable embedding module
- `vector_store/`: ChromaDB persistent files
- `reports/vector_db_specs.md`: Technical documentation

---

## Next Steps

- Implement RAG retriever using LangChain for natural language responses
- Optimize HNSW parameters for faster retrieval
- Add embedding drift detection for ongoing data quality
- Explore GPU acceleration for embedding larger datasets

---

## Contributing

Contributions are welcome! Please submit a pull request or open an issue for bugs, features, or improvements.

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
