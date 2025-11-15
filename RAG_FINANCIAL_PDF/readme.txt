Running the ESG RAG Pipeline

1. Ingest and Embed Data
   Run the following command to chunk all ESG documents and generate their embeddings:
   py ingest_esg_documents.py

2. Launch the Web Inference App
   Start the Streamlit interface to interact with your RAG model:
   streamlit run app_esg.py

3. Run Evaluation
   To evaluate RAG responses against predefined test cases, execute:
   python scripts/run_evaluation.py

   You can modify or add test cases inside the evaluation directory to test how the RAG model responds with different answers.

