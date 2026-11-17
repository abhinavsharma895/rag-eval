# RAG Evaluation using DeepEval, Groq & Gemini

A Python-based RAG evaluation project using **DeepEval** to measure the quality of Retrieval-Augmented Generation (RAG) systems.

This branch integrates **Groq's GPT-OSS 120B** as the LLM evaluation judge and **Google Gemini Embeddings** for embedding-related functionality.

The project evaluates generated answers against expected outputs and retrieved context using multiple evaluation metrics.

## Features

- Evaluate RAG system responses using DeepEval.
- Use Groq GPT-OSS 120B as the evaluation LLM.
- Configure Google Gemini embeddings.
- Evaluate multiple test cases from a JSON file.
- Run individual evaluation metrics.
- Generate evaluation scores and pass/fail results.
- Store evaluation results in JSON format.
- Support synchronous evaluation.

## Tech Stack

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| DeepEval | RAG evaluation framework |
| Groq API | LLM-based evaluation judge |
| GPT-OSS 120B | Evaluation model |
| Google Gemini | Embedding model |
| LangChain Google GenAI | Gemini embeddings integration |
| Pydantic | Structured output validation |
| python-dotenv | Environment variable management |
| JSON | Test case and result storage |

## Project Architecture

```text
              RAG Evaluation Project
                       |
                       v
                Load Test Cases
                   cases.json
                       |
                       v
                 DeepEval Framework
                       |
                       v
                  Evaluation
                       |
              +--------+--------+
              |                 |
              v                 v
        Groq API             Test Data
        GPT-OSS 120B         Retrieval Context
              |                 |
              +--------+--------+
                       |
                       v
                Evaluation Metrics
                       |
          +------------+-------------+
          |            |             |
          v            v             v
      Answer       Faithfulness   Contextual
     Relevancy                    Recall
          |            |             |
          +------------+-------------+
                       |
                       v
                Evaluation Scores
                       |
                       v
             results.json
```

## Evaluation Metrics

### 1. Answer Relevancy

Measures whether the generated answer is relevant to the user's question.

**Example:**

- Question: What is machine learning?
- Answer: Machine learning is a branch of AI that enables systems to learn from data.

A relevant answer should directly address the question.

### 2. Faithfulness

Measures whether the generated answer is supported by the retrieved context.

It helps identify hallucinations and unsupported statements.

**Example:**

Retrieved context:
> Python is a programming language.

Generated answer:
> Python is a programming language used for software development.

The answer is supported by the context.

### 3. Contextual Recall

Measures how much of the information required by the expected answer can be found in the retrieved context.

It helps evaluate the quality of the retrieval process.

## Models Used

### Groq – Evaluation Judge

Model:

```text
openai/gpt-oss-120b
```

Groq is used to generate evaluation judgments for DeepEval metrics.

A custom `GroqJudge` class integrates the Groq API with DeepEval using `DeepEvalBaseLLM`.

It supports structured outputs using Pydantic schemas.

### Google Gemini – Embeddings

Model:

```text
gemini-embedding-001
```

Google Gemini embeddings are initialized using:

```python
from langchain_google_genai import GoogleGenerativeAIEmbeddings

embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001",
    google_api_key=GOOGLE_API_KEY
)
```

**Note:** The current evaluation script initializes the Gemini embedding model, but does not yet pass it into the configured DeepEval metrics.

## Project Structure

```text
rag-eval/
│
├── run_eval.py
│
├── cases.json
│
├── results_abhinav.json
│
├── .env
│
├── .gitignore
│
└── README.md
```

| File | Description |
|---|---|
| `run_eval.py` | Main evaluation script |
| `cases.json` | Contains evaluation test cases |
| `results.json` | Stores evaluation results |
| `.env` | Stores API keys |
| `.gitignore` | Prevents sensitive files from being committed |
| `README.md` | Project documentation |

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/abhinavsharma895/rag-eval.git
```

### 2. Navigate to the Project

```bash
cd rag-eval
```

### 3. Create a Virtual Environment

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scripts\activate
```

### 4. Install Dependencies

```bash
pip install deepeval groq pydantic python-dotenv langchain-google-genai
```

## API Configuration

Create a `.env` file in the project root directory.

```env
GROQ_API_KEY=your_groq_api_key
GOOGLE_API_KEY=your_google_api_key
```

Get your API keys:

- Groq: https://console.groq.com/keys
- Google AI Studio: https://aistudio.google.com/apikey

**Security:** Never upload your `.env` file or expose API keys in GitHub commits.

## Test Case Format

The project uses `cases.json` to store test cases.

Example:

```json
{
    "answer_relevancy": [
        {
            "input": "What is machine learning?",
            "actual_output": "Machine learning is a branch of artificial intelligence.",
            "expected_output": "Machine learning is a subset of artificial intelligence.",
            "retrieval_context": [
                "Machine learning is a subset of artificial intelligence."
            ]
        }
    ],
    "faithfulness": [
        {
            "input": "What is Python?",
            "actual_output": "Python is a programming language.",
            "expected_output": "Python is a programming language.",
            "retrieval_context": [
                "Python is a high-level programming language."
            ]
        }
    ],
    "contextual_recall": [
        {
            "input": "What is deep learning?",
            "actual_output": "Deep learning uses neural networks.",
            "expected_output": "Deep learning is a subset of machine learning that uses neural networks.",
            "retrieval_context": [
                "Deep learning is a subset of machine learning.",
                "Deep learning uses neural networks."
            ]
        }
    ]
}
```

## Run Evaluation

### Run All Metrics

```bash
python run_eval.py
```

### Run Answer Relevancy

```bash
python run_eval.py answer_relevancy
```

### Run Faithfulness

```bash
python run_eval.py faithfulness
```

### Run Contextual Recall

```bash
python run_eval.py contextual_recall
```

## Evaluation Results

After evaluation, results are saved in:

```text
results.json
```

Example result:

```json
[
    {
        "test_case": "test_case_0",
        "metric": "Answer Relevancy",
        "score": 1.0,
        "threshold": 0.7,
        "passed": true,
        "reason": "The answer is relevant to the input question."
    }
]
```

### Understanding the Scores

| Score | Interpretation |
|---|---|
| 1.0 | Excellent |
| 0.8 | Good |
| 0.7 | Meets the configured threshold |
| Below 0.7 | Fails the configured threshold |

The current metric threshold is configured to `0.7`.

## Git Branch Information

This README documents the Groq + Gemini integration branch.

To switch to this branch:

```bash
git checkout your-branch-name
```

To push changes:

```bash
git add .
git commit -m "Update Groq and Gemini evaluation documentation"
git push origin your-branch-name
```

The README in this branch is independent of the README in the `main` branch.

## Future Improvements

- Add Contextual Precision evaluation.
- Integrate Gemini embeddings into embedding-based evaluation workflows.
- Add more RAG evaluation metrics.
- Compare different LLM judges.
- Add evaluation result visualizations.
- Build a Streamlit dashboard for evaluation results.
- Add support for larger test datasets.
- Track evaluation performance across multiple model versions.

