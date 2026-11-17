# RAG Evaluation Framework

A Python-based Retrieval-Augmented Generation (RAG) evaluation framework that measures the quality, relevance, and faithfulness of AI-generated answers using **DeepEval, Google Gemini, LangChain, and ChromaDB**.

The project processes PDF documents, retrieves relevant information, generates answers using Gemini, and evaluates the RAG pipeline using different evaluation metrics.

## Features

- **PDF Ingestion:** Extract text from PDF documents.
- **Text Chunking:** Split documents into smaller overlapping chunks.
- **Google Gemini Embeddings:** Generate vector embeddings for document chunks.
- **ChromaDB:** Store and retrieve document embeddings.
- **Semantic Search:** Retrieve the most relevant document chunks for a question.
- **LLM-Based Answer Generation:** Generate answers using Google Gemini.
- **DeepEval Integration:** Evaluate RAG performance using evaluation metrics.
- **Custom Evaluation Dataset:** Maintain questions and expected answers in JSON.
- **Evaluation Reports:** Store scores, thresholds, pass/fail status, and evaluation reasons in JSON.

## Tech Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| LangChain | RAG application framework |
| Google Gemini | Embeddings and answer generation |
| ChromaDB | Vector database |
| DeepEval | RAG evaluation |
| PyPDF | PDF text extraction |
| JSON | Dataset and evaluation results |
| python-dotenv | Environment variable management |

## Project Architecture

```text
             PDF Document
                  |
                  v
            PDF Extraction
                  |
                  v
             Text Splitting
                  |
                  v
          Gemini Embeddings
                  |
                  v
             ChromaDB
                  |
                  v
           User Question
                  |
                  v
           Semantic Search
                  |
                  v
          Retrieve Top 4 Chunks
                  |
                  v
           Gemini LLM
                  |
                  v
           Generated Answer
                  |
                  v
           DeepEval Metrics
                  |
                  v
          Evaluation Results
```

## Project Structure

```text
rag-eval/
│
├── app.py                 # RAG pipeline implementation
├── build_cases.py         # Generate evaluation test cases
├── run_eval.py            # Execute DeepEval evaluations
├── test_pipeline.py       # Test RAG pipeline
│
├── dataset.json           # Evaluation questions and expected answers
├── cases.json             # Generated evaluation test cases
├── results.json           # Evaluation results
│
├── .env                   # API credentials (not committed)
├── .gitignore             # Git ignored files
└── README.md
```

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/abhinavsharma895/rag-eval.git

cd rag-eval
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate the environment.

**Windows:**

```bash
.venv\Scripts\activate
```

**Linux / macOS:**

```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install langchain langchain-core langchain-google-genai langchain-chroma langchain-text-splitters chromadb pypdf python-dotenv deepeval google-genai
```

### 4. Configure environment variables

Create a `.env` file in the root directory.

```env
GOOGLE_API_KEY=your_google_api_key
```

Replace the placeholder with your actual Google Gemini API key.

**Important:** Never commit your `.env` file or expose API credentials on GitHub.

## Configuration

The project uses Google Gemini models.

### Embedding Model

```python
embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-2-preview"
)
```

### Chat Model

```python
model = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    temperature=0.2,
    max_retries=2
)
```

### Vector Database

ChromaDB stores document embeddings in the `.chroma` directory.

The retriever fetches the four most relevant chunks:

```python
retriever = vector_store.as_retriever(
    search_kwargs={"k": 4}
)
```

## Evaluation Metrics

The project uses DeepEval to evaluate the quality of RAG-generated responses.

### 1. Answer Relevancy

Measures how relevant the generated answer is to the user's question.

**Example:** If the question asks about tuition fees, the answer should discuss tuition fees rather than unrelated information.

### 2. Faithfulness

Measures whether the generated answer is supported by the retrieved context.

**Example:** If the retrieved document says the annual fee is ₹50,000, the answer should not invent a different amount.

### 3. Contextual Recall

Measures whether the retrieved context contains the information needed to produce the expected answer.

### Evaluation Configuration

```python
AnswerRelevancyMetric(
    threshold=0.7,
    model=model,
    include_reason=True,
    async_mode=False
)
```

The configured threshold is 0.7.

A metric passes when its evaluation score meets the configured threshold.

## Running the Project

### Step 1: Configure the PDF path

Open `build_cases.py` and configure your PDF location.

```python
PDF_PATH = "path/to/your/document.pdf"
```

### Step 2: Generate evaluation cases

```bash
python build_cases.py
```

This process:

1. Loads the PDF.
2. Extracts document text.
3. Splits text into chunks.
4. Generates embeddings.
5. Stores embeddings in ChromaDB.
6. Reads questions from `dataset.json`.
7. Retrieves relevant document chunks.
8. Generates answers using Gemini.
9. Saves generated cases in `cases.json`.

### Step 3: Run evaluation

Run all configured metrics:

```bash
python run_eval.py
```

Run an individual metric:

```bash
python run_eval.py answer_relevancy
```

```bash
python run_eval.py faithfulness
```

```bash
python run_eval.py contextual_recall
```

### Step 4: View evaluation results

The evaluation results are saved in:

```text
results_abhinav.json
```

The output includes:

- Test case name
- Metric name
- Evaluation score
- Threshold
- Pass/fail status
- Evaluation reason

## Dataset Format

The `dataset.json` file contains evaluation questions and their expected answers.

Example:

```json
{
    "answer_relevancy": [
        {
            "question": "What is the annual tuition fee?",
            "expected_output": "The annual tuition fee is 50000."
        }
    ],
    "faithfulness": [
        {
            "question": "What documents are required for admission?",
            "expected_output": "Students must submit the required admission documents."
        }
    ],
    "contextual_recall": [
        {
            "question": "What are the admission requirements?",
            "expected_output": "Students must satisfy the admission requirements."
        }
    ]
}
```

Use questions and expected answers that match your PDF document.

## API Credentials and Security

The project uses environment-based API credentials.

- Google API credentials are loaded through environment configuration.
- `.env` is excluded from Git using `.gitignore`.
- API keys should never be hardcoded into source files.
- Do not upload private credentials to public repositories.

Note: This project currently implements API authentication for external model services, not user registration, login, JWT authentication, or OAuth.

## Error Handling and Rate Limits

Google Gemini API requests can sometimes fail because of:

- API quota limits.
- Request rate limits.
- Temporary server errors (503).
- Network connectivity issues.

Recommended practices:

- Use appropriate retry mechanisms.
- Add delays between requests when necessary.
- Monitor API usage and quota.
- Avoid unnecessary repeated embedding generation.
- Run evaluation metrics individually when debugging.

## Future Improvements

- [ ] Add a Streamlit evaluation dashboard.
- [ ] Support multiple PDF documents.
- [ ] Add more DeepEval evaluation metrics.
- [ ] Compare different embedding models.
- [ ] Compare different LLM models.
- [ ] Generate evaluation reports and visualizations.
- [ ] Add experiment tracking and historical comparisons.
- [ ] Add user authentication and role-based access.
- [ ] Add configurable evaluation thresholds.
- [ ] Improve API retry and rate-limit handling.

## Learning Objectives

This project demonstrates practical experience with:

- Retrieval-Augmented Generation (RAG).
- Vector databases and semantic search.
- LLM integration using Google Gemini.
- Document processing and chunking.
- Automated LLM evaluation.
- DeepEval evaluation metrics.
- Python-based AI application development.
