# Dell Laptop Chatbot with RAG and Ollama

A sophisticated chatbot application that combines **Retrieval-Augmented Generation (RAG)** with **Ollama** backend to provide intelligent responses about Dell laptop products. The system uses advanced vector embeddings and semantic search to deliver accurate, context-aware answers about laptop specifications, pricing, and recommendations.

## 🏗️ Architecture Overview

```
┌─────────────────┐    ┌──────────────┐    ┌─────────────────┐
│   Web Frontend  │    │   FastAPI    │    │  Ollama Backend │
│   (HTML/JS)     │───▶│   Server     │───▶│   (Gemma 2B)    │
└─────────────────┘    └──────────────┘    └─────────────────┘
                              │
                              ▼
                       ┌──────────────┐
                       │  RAG System  │
                       │              │
                       │ ┌──────────┐ │
                       │ │  Qdrant  │ │
                       │ │ Vector   │ │
                       │ │   DB     │ │
                       │ └──────────┘ │
                       │              │
                       │ ┌──────────┐ │
                       │ │   E5     │ │
                       │ │Embeddings│ │
                       │ └──────────┘ │
                       └──────────────┘
                              │
                              ▼
                       ┌──────────────┐
                       │   Product    │
                       │   Dataset    │
                       │ (CSV File)   │
                       └──────────────┘
```

## 🤖 What is RAG (Retrieval-Augmented Generation)?

RAG is a powerful AI architecture that combines the strengths of:

1. **Information Retrieval**: Finding relevant documents from a knowledge base
2. **Text Generation**: Using a Large Language Model to generate contextual responses

### How RAG Works in This Project:

1. **Document Ingestion**: Dell laptop data (specs, prices, models) is loaded from CSV
2. **Vectorization**: Product information is converted to high-dimensional vectors using E5 embeddings
3. **Storage**: Vectors are stored in Qdrant vector database for efficient similarity search
4. **Query Processing**: User questions are converted to vectors and matched against the database
5. **Context Retrieval**: Most relevant products are retrieved based on semantic similarity
6. **Response Generation**: Ollama LLM generates natural language responses using retrieved context

### RAG Components in the Project:

```python
# Vector Embeddings Model
EMB_MODEL = "intfloat/e5-base-v2"

# Vector Database
QDRANT_URL = "https://your-qdrant-instance.com"
QDRANT_COLLECTION = "laptops"

# Retrieval System
class ProductRetrieverQdrant:
    - Semantic search using vector similarity
    - Price range filtering
    - Brand filtering (Dell focus)
    - Fallback keyword search
```

## 🦙 Ollama Backend

**Ollama** is a lightweight framework for running Large Language Models locally. This project uses Ollama to power the conversational AI component.

### Why Ollama?

- **Local Deployment**: Run LLMs without cloud dependencies
- **Multiple Model Support**: Easy model switching (Gemma, Llama, etc.)
- **Resource Efficient**: Optimized for various hardware configurations
- **Privacy**: All processing happens locally/on-premises

### Ollama Configuration:

```python
# Model Settings
MODEL_SERVER = "http://127.0.0.1:11434"  # Local Ollama server
MODEL_NAME = "gemma2:2b-instruct"        # Efficient 2B parameter model

# Generation Parameters
TEMP = 0.2          # Low temperature for consistent responses
MAX_TOKENS = 256    # Concise responses
TOP_P = 0.9         # Nucleus sampling
```

### Supported Models:
- **Gemma 2B**: Lightweight, efficient model (default)
- **Llama 3.1**: More capable but resource-intensive
- **Custom Models**: Easy to switch via environment variables

## 🚀 Key Features

### Intelligent Product Search
- **Semantic Understanding**: Understands intent beyond keyword matching
- **Multi-criteria Filtering**: Price range, brand, specifications
- **Relevance Scoring**: Returns most appropriate products first

### Price Intelligence
- **Multi-currency Support**: INR, USD, CAD, HKD
- **Market Localization**: Different pricing for different markets
- **Tax Calculations**: Automatic tax inclusion based on region

### Conversation Management
- **Context Awareness**: Remembers conversation context
- **Follow-up Questions**: Handles clarifications and confirmations
- **Greeting Recognition**: Natural conversation flow

### Robust Fallback System
- **Offline Mode**: Works without vector database connection
- **Keyword Search**: Falls back to text-based search when needed
- **Error Handling**: Graceful degradation of features

## 📁 Project Structure

```
chatbot/
├── app/
│   ├── app.py                 # Main FastAPI application
│   ├── llm/
│   │   └── compose.py         # Ollama LLM integration
│   ├── rag/
│   │   ├── retriever_qdrant.py # Vector search implementation
│   │   ├── retriever_tfidf.py  # TF-IDF fallback search
│   │   ├── helper.py          # Utility functions
│   │   └── policy.py          # Business logic
│   ├── static/               # Frontend assets
│   ├── templates/            # HTML templates
│   └── laptops_norm.csv      # Product dataset
├── ollama-backend/
│   └── Dockerfile           # Ollama container configuration
├── requirements.txt         # Python dependencies
├── Dockerfile              # Main app container
└── .env                    # Environment configuration
```

## 🛠️ Technology Stack

### Backend
- **FastAPI**: Modern Python web framework
- **Pandas**: Data manipulation and analysis
- **Sentence Transformers**: Text embeddings (E5 model)
- **Qdrant**: Vector database for semantic search
- **Ollama**: Local LLM serving

### Frontend
- **HTML5/CSS3**: Modern web interface
- **JavaScript**: Interactive chat functionality
- **Responsive Design**: Mobile-friendly layout

### Infrastructure
- **Docker**: Containerized deployment
- **Google Cloud Run**: Scalable cloud hosting
- **Environment Variables**: Configuration management

## ⚙️ Configuration

### Core Settings
```bash
# LLM Configuration
MODEL_SERVER=http://127.0.0.1:11434
MODEL_NAME=gemma2:2b-instruct
LLM_TEMPERATURE=0.2
LLM_MAX_TOKENS=256

# RAG Configuration
QDRANT_URL=https://your-qdrant-instance.com
EMB_MODEL_PATH=intfloat/e5-base-v2
RELEVANCE_THRESHOLD=0.12

# Business Logic
ONLY_DELL=1
DEFAULT_MARKET=HK
PRICE_CURRENCY=INR
```

### Market & Currency Support
```bash
# Multi-currency pricing
FX_INR_TO_CAD=0.016
FX_INR_TO_HKD=0.094
FX_USD_TO_CAD=1.36

# Regional tax rates
CA_PROVINCE=ON  # Ontario tax: 13%
INCLUDE_TAX_LINE=1
```

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Set Up Ollama
```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Pull the model
ollama pull gemma2:2b-instruct

# Start Ollama server
ollama serve
```

### 3. Configure Vector Database
```bash
# Set up Qdrant (optional - has CSV fallback)
export QDRANT_URL="your-qdrant-url"
export QDRANT_API_KEY="your-api-key"
```

### 4. Run the Application
```bash
cd app
uvicorn app:app --host 0.0.0.0 --port 8000
```

### 5. Access the Chatbot
Open your browser and navigate to `http://localhost:8000`

## 📊 Data Pipeline

### 1. Data Ingestion
```python
# Load laptop dataset
df = pd.read_csv("laptops_norm.csv")

# Create searchable text
df["_search_text"] = (
    df["brand"] + " " + 
    df["model"] + " " + 
    df["Specification"]
)
```

### 2. Vector Processing
```python
# Generate embeddings
model = SentenceTransformer("intfloat/e5-base-v2")
vectors = model.encode([f"passage: {text}" for text in df["_search_text"]])

# Store in Qdrant
client.upsert(collection_name="laptops", points=vectors)
```

### 3. Query Processing
```python
# Convert query to vector
query_vector = model.encode([f"query: {user_question}"])

# Semantic search
results = client.search(
    collection_name="laptops",
    query_vector=query_vector,
    limit=10
)
```

## 🔧 Advanced Features

### Smart Price Queries
```python
# Natural language price understanding
"laptops under $1000" → price_max=1000, currency=USD
"Dell laptops between 50000 to 80000 INR" → brand=Dell, price_range=[50000,80000]
```

### Context-Aware Responses
```python
# RAG prompt engineering
prompt = f"""You are a concise laptop expert. Use ONLY the FACTS section.

USER QUERY: {user_query}

FACTS: {retrieved_context}

GUIDELINES:
- Use only information in FACTS
- If detail missing, say "not available"
- Answer in <= 120 words
"""
```

### Conversation State Management
```python
# Handle follow-up questions
if should_confirm(query):
    LAST_CONFIRM_NEXT = query
    return "Would you like me to show you specific Dell models?"

# Process confirmations
if user_input in YES_WORDS and LAST_CONFIRM_NEXT:
    # Execute the pending query
    results = retriever.search(LAST_CONFIRM_NEXT)
```

## 📈 Performance Optimizations

### Vector Search
- **E5 Base v2**: Balanced performance/quality embedding model
- **DOT Product**: Efficient similarity computation
- **Filtered Search**: Pre-filtering by brand/price before vector comparison

### LLM Optimization
- **Small Model**: Gemma 2B for faster inference
- **Low Temperature**: Consistent, factual responses
- **Token Limits**: Concise responses for better UX

### Caching Strategy
- **Model Persistence**: Keep Ollama model loaded in memory
- **Connection Pooling**: Reuse Qdrant connections
- **Fallback Modes**: CSV search when vector DB unavailable

## 🔒 Security & Privacy

### Data Privacy
- **Local Processing**: LLM runs locally via Ollama
- **No External APIs**: No data sent to third-party services
- **Secure Storage**: Environment variables for sensitive config

### Access Control
- **Bearer Token Support**: Optional API authentication
- **CORS Configuration**: Controlled cross-origin requests
- **Input Validation**: Sanitized user inputs

## 📝 Example Conversations

### Product Inquiry
```
User: "Show me Dell gaming laptops under $1500"
Bot: "Dell G15 Gaming Laptop
      Price: $1,299 USD (+ tax)
      Specs: Intel i7-11800H, RTX 3060, 16GB RAM, 512GB SSD
      
      Dell Inspiron 16 Plus
      Price: $1,449 USD (+ tax)  
      Specs: Intel i7-12700H, RTX 3050, 16GB RAM, 1TB SSD"
```

### Specification Comparison
```
User: "Compare Dell XPS 13 vs Dell Inspiron 14"
Bot: "Dell XPS 13: Premium ultrabook, Intel i7-1260P, 16GB RAM, 512GB SSD - $1,199
      Dell Inspiron 14: Budget-friendly, Intel i5-1235U, 8GB RAM, 256GB SSD - $649
      
      Key differences: XPS has better build quality, faster processor, more RAM"
```

## 🤝 Contributing

### Development Setup
1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make changes and test thoroughly
4. Submit a pull request with detailed description

### Adding New Features
- **New Retrievers**: Implement in `app/rag/`
- **LLM Models**: Configure in `app/llm/compose.py`
- **Business Logic**: Extend `app/rag/helper.py`

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Ollama Team**: For the excellent local LLM framework
- **Qdrant**: For the high-performance vector database
- **Sentence Transformers**: For the E5 embedding model
- **FastAPI**: For the modern Python web framework

---

*Built with ❤️ using RAG + Ollama for intelligent laptop recommendations*