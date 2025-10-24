# Financial ESG RAG System

Retrieval Augmented Generation (RAG) system for analyzing ESG financial reports using Qdrant vector database, conversation memory, and Ollama LLM.

## Overview

This project implements a production-ready RAG system that allows users to query ESG (Environmental, Social, Governance) financial reports from multiple South African companies. The system uses semantic search to retrieve relevant information and generates accurate, contextual answers with proper source citations.

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Ingest Documents
```bash
python scripts/ingest_esg_documents.py --clear
```

### 3. Run Web Interface
```bash
streamlit run app_esg.py
```
The application will open at `http://localhost:8501`

## Project Structure

```
├── src/
│   ├── chatbot/esg_chatbot.py    # Main chatbot orchestrator
│   ├── rag/qdrant_store.py        # Vector database interface
│   ├── memory/                    # Conversation history management
│   ├── llm/ollama_client.py       # LLM client interface
│   └── evaluation/                # Metrics and testing framework
├── scripts/
│   ├── ingest_esg_documents.py    # PDF document ingestion pipeline
│   └── run_evaluation.py          # RAG system evaluation
├── app_esg.py                      # Streamlit web interface
└── data/                           # ESG PDF reports and databases
```

## Configuration

Create a `.env` file in the project root:
```env
OLLAMA_API_URL=https://your-ollama-endpoint.com
OLLAMA_MODEL=gemma3:4b
QDRANT_COLLECTION_NAME=esg_financial_reports
MAX_CONVERSATION_HISTORY=20
```

## Dataset

The system processes real ESG financial reports from 7 major South African companies. The dataset consists of publicly available PDF reports totaling approximately 50MB of financial and sustainability data.

### Document Statistics
- Total Documents: 7 PDF reports
- Total Size: ~50 MB
- Total Pages: ~400+ pages
- Vector Database: 2,757 document chunks
- Companies: 7 South African corporations
- Report Years: 2021-2023

### Companies and Coverage

- **Absa Group (2022)** - 254 KB
  Energy consumption metrics, carbon emissions data, renewable energy targets, water usage

- **Sasol (2023)** - 17.3 MB
  Employee training programs, sustainability initiatives, GHG emissions, net-zero targets

- **Clicks (2022)** - 3.0 MB
  Carbon neutral targets by 2030, ESG performance metrics, waste diversion rates

- **Pick n Pay (2023)** - 2.1 MB
  ESG goals, sustainability programs, packaging reduction, renewable energy targets

- **Distell (2022)** - 1.1 MB
  Water consumption (4.2M m³), carbon footprint, ESG reporting metrics

- **Tongaat Hulett (2021)** - 15.0 MB
  Carbon emissions and reduction initiatives, board diversity (20% female representation)

- **Implats (2023)** - 11.5 MB
  Scope 1 and 2 GHG emissions, sustainability frameworks (GRI, TCFD, UNGC, SDGs)

### Data Categories

The dataset covers key ESG metrics including:
- **Environmental**: Carbon emissions, energy consumption, water usage, waste management
- **Social**: Employee metrics, board diversity, training programs, community impact
- **Governance**: Sustainability frameworks, reporting standards, corporate policies

### Evaluation Test Dataset

A curated test dataset of 22 queries with ground truth answers was created for evaluation:
- 13 standard test cases covering metrics, targets, and general ESG questions
- 6 adversarial test cases for robustness testing
- 3 cross-company comparison queries

Test categories:
- Metrics queries (specific numbers and KPIs)
- Target queries (goals and commitments)
- General ESG questions (frameworks and principles)
- Comparison queries (multi-company analysis)
- Off-topic detection
- Missing data handling

## Technical Features

- Vector database using Qdrant for semantic search
- SQLite-based conversation memory for context retention
- Dual query processing for numerical and textual queries
- Source citation with page numbers and confidence scores
- Streamlit web interface for easy interaction
- Comprehensive evaluation framework with 11 metrics
- Modular architecture for easy extension


## Technologies Used

- Python 3.8+
- Qdrant (Vector Database)
- Ollama (LLM)
- Streamlit (Web Interface)
- LangChain (Document Processing)
- Sentence Transformers (Embeddings)
- SQLite (Conversation Memory)

## Evaluation Results

The system was evaluated using 22 test queries with ground truth answers across multiple ESG categories.

### Performance Metrics

| Metric | Score | Description |
|--------|-------|-------------|
| Overall Score | 0.825 | Combined performance |
| Answer Relevancy | 0.841 | Relevance to questions |
| Company Accuracy | 0.955 | Correct company identification |
| Context Precision | 0.909 | Retrieved chunk quality |
| Context Recall | 0.941 | Information coverage |
| Faithfulness | 0.860 | Factual accuracy |
| MRR | 0.865 | Mean Reciprocal Rank |
| NDCG@5 | 1.942 | Ranking quality |
| Numerical Accuracy | 0.594 | Numerical data accuracy |
| Source Citation | 0.647 | Source reference quality |

### Test Coverage
- Total Test Queries: 22
- Success Rate: 100% (22/22)
- Categories Tested: Metrics, targets, general ESG, comparisons, adversarial cases

### Running Evaluation
```bash
python scripts/run_evaluation.py
```

