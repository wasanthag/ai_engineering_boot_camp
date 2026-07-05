# Week 1 Study Guide: RAG Systems Fundamentals

> **Bootcamp Focus:** Building production-ready RAG (Retrieval-Augmented Generation) systems
>
> **Use Case:** Amazon Electronics Product Recommendation Chatbot

---

## Quick Reference Index

0. [Setup & Prerequisites](#0-setup--prerequisites)
   - [LangSmith Account](#langsmith-account-setup)
   - [Development Environment with uv](#development-environment-with-uv)
1. [Data Preprocessing](#1-data-preprocessing)
2. [Vector Embeddings & Search](#2-vector-embeddings--search)
3. [RAG Pipeline](#3-rag-pipeline)
4. [Observability](#4-observability)
5. [Evaluation Dataset](#5-evaluation-dataset)
6. [RAG Evaluation Metrics](#6-rag-evaluation-metrics)
7. [Production Deployment](#7-production-deployment)

---

## 0. Setup & Prerequisites

### LangSmith Account Setup

**What:** LangSmith is Anthropic's platform for LLM observability, tracing, and evaluation

**Why You Need It:**
- Store and visualize execution traces from your RAG pipeline
- Create and manage evaluation datasets
- Run experiments and compare RAG system versions
- Track token usage and costs

#### Creating Your Free Account

1. **Sign Up:**
   - Visit: https://smith.langchain.com/
   - Click "Sign Up" and create a free account
   - Verify your email

2. **Get Your API Key:**
   - Navigate to Settings → API Keys
   - Click "Create API Key"
   - Copy the key (starts with `lsv2_pt_...`)

3. **Add to Environment Variables:**

   Edit your `.env` file:
   ```bash
   # LangSmith Configuration
   LANGSMITH_API_KEY=lsv2_pt_your_key_here
   LANGCHAIN_TRACING_V2=true
   LANGCHAIN_PROJECT=ai-engineering-bootcamp-week1
   ```

4. **Verify Setup:**
   ```python
   from langsmith import Client

   ls_client = Client()
   print("LangSmith connected!")
   # If this works, you're ready to go
   ```

#### What You'll Use LangSmith For

**In Notebook 4 (Observability):**
- View execution traces for every RAG pipeline run
- See token usage per component (embedding, retrieval, generation)
- Debug slow requests or errors

**In Notebook 5 (Eval Dataset):**
- Store synthetic evaluation datasets
- Add ground truth answers
- Version your evaluation data

**In Notebook 6 (Evaluations):**
- Run RAGAS metrics on entire datasets
- Compare experiments (e.g., different prompt templates)
- Track metric trends over time

**In Production:**
- Monitor live traffic
- A/B test different RAG configurations
- Catch quality regressions

---

### Development Environment with uv

**What:** `uv` is a blazingly fast Python package manager written in Rust

**Why Use uv Instead of pip/venv:**
- 10-100x faster than pip
- Built-in virtual environment management
- Workspace support for monorepos
- Lockfile for reproducible builds (`uv.lock`)
- Automatic dependency resolution

#### Installation

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Or with Homebrew
brew install uv

# Verify installation
uv --version
```

#### Project Structure with uv

This bootcamp uses a **workspace** setup:

```
ai_engineering_boot_camp/
├── pyproject.toml          # Root project config
├── uv.lock                 # Lockfile (like package-lock.json)
├── .venv/                  # Virtual environment
├── apps/
│   ├── api/
│   │   └── pyproject.toml  # API-specific dependencies
│   └── chatbot_ui/
│       └── pyproject.toml  # UI-specific dependencies
└── notebooks/
```

**Key File:** `pyproject.toml`

```toml
[project]
name = "ai-engineering-bootcamp-prerequisites"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = [
    "google-genai>=1.57.0",
    "openai>=2.15.0",
    "streamlit>=1.52.2",
    # Base dependencies for all projects
]

[tool.uv.workspace]
members = ["apps/api", "apps/chatbot_ui"]  # Monorepo setup

[dependency-groups]
dev = [
    "langchain-openai>=1.1.9",
    "langsmith>=0.8.16",
    "matplotlib>=3.11.0",
    "qdrant-client>=1.18.0",
    "ragas>=0.4.3",
    # Development/notebook-specific packages
]
```

#### Basic uv Commands

**1. Create Virtual Environment**

```bash
# Create default .venv
uv venv

# Create named environment
uv venv dev

# Activate (same as regular venv)
source .venv/bin/activate      # macOS/Linux
.venv\Scripts\activate         # Windows
```

**2. Install Dependencies**

```bash
# Install from pyproject.toml
uv sync

# Install with dev dependencies
uv sync --group dev

# Install all groups (dev + any others)
uv sync --all-groups

# Install a new package and add to pyproject.toml
uv add openai

# Install to dev group
uv add --group dev jupyter

# Install specific version
uv add "pandas>=2.0.0,<3.0.0"
```

**3. Workspace Commands**

```bash
# Sync entire workspace (all members)
uv sync --workspace

# Add package to specific workspace member
uv add --project apps/api fastapi

# Run command in workspace member
uv run --project apps/api python -m api.app
```

**4. Managing Dependencies**

```bash
# Update all packages to latest compatible versions
uv sync --upgrade

# Update specific package
uv add --upgrade openai

# Remove package
uv remove matplotlib

# Remove from dev group
uv remove --group dev jupyter

# Show installed packages
uv pip list

# Show dependency tree
uv tree
```

#### Typical Workflow for This Bootcamp

**Initial Setup:**
```bash
# 1. Clone repository
git clone <repo-url>
cd ai_engineering_boot_camp

# 2. Create virtual environment
uv venv

# 3. Activate
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

# 4. Install all dependencies (base + dev)
uv sync --group dev

# 5. Verify
python -c "import openai, langsmith, qdrant_client; print('All packages installed!')"
```

**Adding a New Package (Example: Adding pandas):**
```bash
# For notebooks/development
uv add --group dev pandas

# This updates pyproject.toml:
# [dependency-groups]
# dev = [
#     ...
#     "pandas>=2.2.0",
# ]

# And automatically installs pandas
```

**Syncing from pyproject.toml (After Pulling Changes):**
```bash
# Someone else added new dependencies and pushed pyproject.toml
git pull

# Sync your environment to match
uv sync --group dev

# This reads pyproject.toml and uv.lock
# Installs any missing packages
# Removes packages no longer listed
```

#### uv vs pip Comparison

| Task | pip | uv |
|------|-----|-----|
| Create venv | `python -m venv .venv` | `uv venv` |
| Install deps | `pip install -r requirements.txt` | `uv sync` |
| Add package | `pip install pandas` | `uv add pandas` |
| Update all | `pip install --upgrade -r requirements.txt` | `uv sync --upgrade` |
| Lock deps | Manual `pip freeze` | Automatic `uv.lock` |
| Speed | Baseline | 10-100x faster |

#### Dependency Groups Explained

This project uses **dependency groups** to separate concerns:

**Base Dependencies (project.dependencies):**
- Needed for ALL environments
- Example: `openai`, `pydantic`, `streamlit`
- Always installed with `uv sync`

**Dev Group (dependency-groups.dev):**
- Only for development/notebooks
- Example: `jupyter`, `matplotlib`, `langsmith`, `ragas`
- Install with `uv sync --group dev`

**Why This Matters:**
```bash
# Production deployment (Docker)
uv sync  # Only base dependencies, smaller image

# Local development
uv sync --group dev  # Everything including notebooks
```

#### Workspace Benefits

The workspace setup allows:

1. **Shared dependencies:** Common packages in root `pyproject.toml`
2. **Isolated apps:** Each app has its own dependencies
3. **Single lockfile:** One `uv.lock` for entire monorepo
4. **Faster installs:** Shared cache across workspace

**Example:**
```bash
# Install everything at once
uv sync --workspace --group dev

# This installs:
# - Root dependencies (openai, pydantic, etc.)
# - apps/api dependencies (fastapi, uvicorn)
# - apps/chatbot_ui dependencies (streamlit)
# - Dev dependencies (jupyter, langsmith, ragas)
```

#### Common uv Commands Cheat Sheet

```bash
# Environment
uv venv                     # Create .venv
uv venv myenv               # Create named env

# Install
uv sync                     # Install from pyproject.toml
uv sync --group dev         # Include dev group
uv sync --all-groups        # Include all groups
uv sync --workspace         # Sync entire workspace

# Add packages
uv add requests             # Add to main dependencies
uv add --group dev pytest   # Add to dev group
uv add "django>=4.0"        # Add with version constraint

# Update
uv sync --upgrade           # Update all packages
uv add --upgrade openai     # Update specific package

# Remove
uv remove pandas            # Remove from main deps
uv remove --group dev pytest # Remove from dev group

# Info
uv pip list                 # List installed packages
uv tree                     # Show dependency tree
uv pip show openai          # Package details

# Run
uv run python script.py     # Run with uv-managed Python
uv run jupyter notebook     # Run Jupyter
```

#### Troubleshooting

**Issue: Package not found after `uv add`**
```bash
# Make sure environment is activated
source .venv/bin/activate

# Or use uv run
uv run python -c "import package_name"
```

**Issue: Dependency conflicts**
```bash
# Clear cache and reinstall
rm -rf .venv uv.lock
uv venv
uv sync --group dev
```

**Issue: Different versions on team members' machines**
```bash
# Always commit uv.lock to git
git add uv.lock pyproject.toml
git commit -m "Update dependencies"

# Team members sync from lockfile
git pull
uv sync --group dev  # Installs exact versions from uv.lock
```

---

## 1. Data Preprocessing

### Concept
**What:** Transforming raw data into clean, structured format suitable for RAG systems

**Why:** Raw data is often messy, incomplete, or contains irrelevant information that degrades retrieval quality

### Key Techniques

#### Data Filtering
```python
# Filter by date
def filter_data_by_date(data: dict) -> bool:
    return int(data["details"]["Date First Available"][-4:]) < 2022

# Filter by rating count
df_ratings_100 = df[df["rating_number"] >= 100]
```

**Purpose:** Focus on recent, well-reviewed products for better recommendations

#### Sampling Strategy
```python
# Stratified sampling for balanced dataset
df_sample = df_ratings_100.sample(n=1000, random_state=42)
```

**Purpose:** Reduce dataset size while maintaining statistical diversity

### In This Project

**Original Dataset:** 1.6M+ Amazon Electronics products
**After Date Filter:** 103,993 products (2022-2023)
**After Rating Filter:** 17,290 products (100+ ratings)
**Final Sample:** 1,000 products → 50 for development

**Files:** `notebooks/week_1/01-explore-amazon-dataset.ipynb`

---

## 2. Vector Embeddings & Search

### Concept
**What:** Converting text into numerical vectors that capture semantic meaning

**Why:** Enables similarity-based retrieval instead of keyword matching

### The Embedding Model

```python
model = "text-embedding-3-small"
dimensions = 1536
provider = "OpenAI"

response = openai.embeddings.create(
    input=text,
    model="text-embedding-3-small"
)
embedding = response.data[0].embedding  # [0.123, -0.456, ...]
```

### Vector Database: Qdrant

#### Setup
```python
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

client = QdrantClient(url="http://localhost:6333")

# Create collection
client.create_collection(
    collection_name="Amazon-items-collection-01",
    vectors_config=VectorParams(
        size=1536,           # Match embedding dimensions
        distance=Distance.COSINE  # Similarity metric
    )
)
```

#### Inserting Data
```python
from qdrant_client.models import PointStruct

points = [
    PointStruct(
        id=i,
        vector=get_embedding(item["preprocessed_description"]),
        payload={
            "preprocessed_description": item["description"],
            "image": item["image"],
            "rating_number": item["rating_number"],
            "price": item["price"],
            "average_rating": item["average_rating"],
            "parent_asin": item["parent_asin"]  # Unique product ID
        }
    )
    for i, item in enumerate(data)
]

client.upsert(collection_name="Amazon-items-collection-01", points=points)
```

#### Retrieval
```python
def retrieve_data(query, k=5):
    query_embedding = get_embedding(query)
    results = client.query_points(
        collection_name="Amazon-items-collection-01",
        query=query_embedding,
        limit=k
    )
    return results
```

### Text Preprocessing Strategy

```python
# Combine title and features for rich semantic content
preprocessed_description = f"{title} {' '.join(features)}"
```

**Example:**
```
Title: "Bluetooth Beanie Hat"
Features: ["100% Acrylic", "Built-in Speakers", "10hr Battery"]
→ "Bluetooth Beanie Hat 100% Acrylic Built-in Speakers 10hr Battery"
```

**Files:** `notebooks/week_1/02-RAG-preprocessing-items.ipynb`

---

## 3. RAG Pipeline

### Concept
**What:** Retrieval-Augmented Generation - combining search with LLM generation

**Why:** Grounds LLM responses in actual data, reducing hallucinations

### The 4-Step RAG Pipeline

```
User Query → [1] Retrieval → [2] Context Processing → [3] Prompt Building → [4] Generation
```

#### Step 1: Retrieval
```python
def retrieve_data(query, k=5):
    query_embedding = get_embedding(query)
    results = qdrant_client.query_points(
        collection_name="Amazon-items-collection-01",
        query=query_embedding,
        limit=k
    )
    return results
```

**Output:** Top-k most similar products with scores

#### Step 2: Context Processing
```python
def process_context(context):
    chunks = []
    for point in context.points:
        chunk = (
            f"- ID: {point.payload['parent_asin']}, "
            f"rating: {point.payload['average_rating']}, "
            f"description: {point.payload['preprocessed_description']}\n"
        )
        chunks.append(chunk)
    return "".join(chunks)
```

**Purpose:** Format retrieved data into readable context for the LLM

#### Step 3: Prompt Building
```python
SYSTEM_INSTRUCTIONS = """
You are a helpful shopping assistant.
Answer questions only based on the provided product information.
Never use the word 'context' - refer to 'available products' instead.
Do not use markdown formatting in your response.
"""

def build_prompt(context, question):
    return f"{SYSTEM_INSTRUCTIONS}\n\n{context}\n\nQuestion: {question}"
```

#### Step 4: Answer Generation
```python
def generate_answer(prompt):
    response = openai.chat.completions.create(
        model="gpt-5.4-nano",
        messages=[{"role": "user", "content": prompt}],
        reasoning_effort="none"  # Faster responses
    )
    return response.choices[0].message.content
```

### Complete Pipeline
```python
def rag_pipeline(question, top_k=5):
    # 1. Retrieve
    retrieved_context = retrieve_data(question, k=top_k)

    # 2. Process
    preprocessed_context = process_context(retrieved_context)

    # 3. Build prompt
    prompt = build_prompt(preprocessed_context, question)

    # 4. Generate
    answer = generate_answer(prompt)

    return answer
```

### Example Query Flow

**Input:** "Do you have a USB fan for hot summers?"

**Retrieval Results:**
```
- ID: B0BRJS644Z, rating: 4.7
  description: "Marame 120mm USB Powered Fan with Speed Controller..."
- ID: B099N9F3FP, rating: 4.2
  description: "AICHESON Laptop Cooler with 6 Cooling Fans..."
```

**Generated Answer:**
```
Yes, we have USB-powered cooling fans. The Marame 120mm USB fan (rating 4.7)
features speed control and works with 5V USB power...
```

**Files:** `notebooks/week_1/03-RAG-pipeline.ipynb`

---

## 4. Observability

### Concept
**What:** Tracking and monitoring RAG pipeline execution in production

**Why:** Debug issues, optimize performance, track costs

### LangSmith Tracing

#### The `@traceable` Decorator
```python
from langsmith import traceable

@traceable(
    name="function_name",
    run_type="embedding|retriever|llm|prompt",
    metadata={
        "ls_provider": "openai",
        "ls_model_name": "gpt-5.4-nano"
    }
)
def my_function():
    pass
```

### Instrumented Pipeline

#### 1. Embedding Function
```python
@traceable(
    name="embed_query",
    run_type="embedding",
    metadata={
        "ls_provider": "openai",
        "ls_model_name": "text-embedding-3-small"
    }
)
def get_embedding(text, model="text-embedding-3-small"):
    response = openai.embeddings.create(input=text, model=model)

    # Track token usage
    run = get_current_run_tree()
    run.outputs = {
        "input_tokens": response.usage.prompt_tokens,
        "total_tokens": response.usage.total_tokens
    }

    return response.data[0].embedding
```

#### 2. Retriever
```python
@traceable(name="retrieve_data", run_type="retriever")
def retrieve_data(query, k=5):
    query_embedding = get_embedding(query)
    results = qdrant_client.query_points(
        collection_name="Amazon-items-collection-01",
        query=query_embedding,
        limit=k
    )
    return results
```

#### 3. Context Formatter
```python
@traceable(name="format_retrieved_context", run_type="prompt")
def process_context(context):
    # Format context for LLM
    return formatted_string
```

#### 4. LLM Generator
```python
@traceable(
    name="generate_answer",
    run_type="llm",
    metadata={
        "ls_provider": "openai",
        "ls_model_name": "gpt-5.4-nano"
    }
)
def generate_answer(prompt):
    response = openai.chat.completions.create(
        model="gpt-5.4-nano",
        messages=[{"role": "user", "content": prompt}],
        reasoning_effort="none"
    )

    # Track token usage
    run = get_current_run_tree()
    run.outputs = {
        "input_tokens": response.usage.prompt_tokens,
        "output_tokens": response.usage.completion_tokens,
        "total_tokens": response.usage.total_tokens
    }

    return response.choices[0].message.content
```

#### 5. Full Pipeline
```python
@traceable(name="rag_pipeline")
def rag_pipeline(question, top_k=5):
    retrieved_context = retrieve_data(question, k=top_k)
    preprocessed_context = process_context(retrieved_context)
    prompt = build_prompt(preprocessed_context, question)
    answer = generate_answer(prompt)
    return answer
```

### What You Can Track

- **Execution Flow:** Call hierarchy and timing
- **Token Usage:** Costs per request
- **Inputs/Outputs:** Debug retrieval and generation
- **Errors:** Stack traces and failures
- **Performance:** Latency per component

**LangSmith Dashboard:** View traces at https://smith.langchain.com

**Files:** `notebooks/week_1/04-Observability-foundations.ipynb`

---

## 5. Evaluation Dataset

### Concept
**What:** Synthetic dataset of questions with ground truth answers

**Why:** Measure RAG system quality systematically

### Synthetic Generation with LLMs

#### The Strategy

Use an LLM to generate realistic questions from your actual data:

```python
# Get all product contexts
all_contexts = [
    {"id": product.payload["parent_asin"],
     "text": product.payload["preprocessed_description"]}
    for product in qdrant_client.scroll(collection_name="...")[0]
]
```

#### LLM Prompt Design

```python
SYSTEM_PROMPT = """
You are building a RAG evaluation dataset for a shopping assistant.

Given 50 product descriptions, generate 30 questions:
- 10 questions requiring MULTIPLE products to answer
- 15 questions requiring SINGLE product to answer
- 5 questions that CANNOT be answered from the products

For each question, provide:
1. reasoning: Why you chose these products
2. question: Natural user question
3. chunk_ids: Product IDs needed to answer
4. answer_example: Ground truth answer
"""

response = openai.chat.completions.create(
    model="gpt-5.4",  # More capable model for generation
    messages=[
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(all_contexts)}
    ],
    reasoning_effort="none"
)
```

#### Output Schema

```json
[
  {
    "reasoning": "This question can be answered using the Bluetooth speaker product",
    "question": "Do you have any Bluetooth speakers with long battery life?",
    "chunk_ids": ["B0C996WY16"],
    "answer_example": "Yes, we have the Raymate Bluetooth Speaker with 30-hour playtime..."
  },
  {
    "reasoning": "This requires comparing multiple cooling fan products",
    "question": "Which USB fans have the best ratings?",
    "chunk_ids": ["B0BRJS644Z", "B099N9F3FP"],
    "answer_example": "The Marame 120mm fan has 4.7 rating while..."
  },
  {
    "reasoning": "Cannot be answered - no laptop information",
    "question": "Do you sell gaming laptops?",
    "chunk_ids": [],
    "answer_example": "We don't have information about gaming laptops."
  }
]
```

### Storing in LangSmith

```python
from langsmith import Client

ls_client = Client()

# Create dataset
dataset = ls_client.create_dataset(
    dataset_name="rag-evaluation-dataset",
    description="RAG evaluation dataset"
)

# Add examples
for item in synthetic_data:
    ls_client.create_example(
        dataset_id=dataset.id,
        inputs={"question": item["question"]},
        outputs={
            "ground_truth": item["answer_example"],
            "reference_context_ids": item["chunk_ids"],
            "reference_descriptions": [
                get_description(id) for id in item["chunk_ids"]
            ]
        }
    )
```

### Quality Distribution

**Generated Dataset:**
- ✅ 15 single-chunk questions
- ✅ 12 multi-chunk questions
- ✅ 5 impossible questions
- **Total:** 32 evaluation examples

**Files:** `notebooks/week_1/05-RAG-Eval-Dataset.ipynb`

---

## 6. RAG Evaluation Metrics

### Concept
**What:** Quantitative measures of RAG system quality

**Why:** Track improvements, catch regressions, compare approaches

### The RAGAS Framework

**RAGAS** = Retrieval-Augmented Generation Assessment

#### Setup
```python
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# Configure for RAGAS
ragas_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-5.4-mini"))
ragas_embeddings = LangchainEmbeddingsWrapper(
    OpenAIEmbeddings(model="text-embedding-3-small")
)
```

### The 4 Core Metrics

#### Metric 1: Context Precision (Retrieval Quality)

**Question:** Are the retrieved chunks relevant?

```python
from ragas.metrics import IDBasedContextPrecision

metric = IDBasedContextPrecision()

# Compares
retrieved_ids = ["B0BRJS644Z", "B099N9F3FP", "B0BZJKX7MS"]
reference_ids = ["B0BRJS644Z", "B099N9F3FP"]

# Score: 2 relevant / 3 retrieved = 0.67
```

**Interpretation:**
- **1.0** = Perfect retrieval (all retrieved chunks are relevant)
- **0.5** = Half retrieved chunks are noise
- **0.0** = No relevant chunks retrieved

#### Metric 2: Context Recall (Coverage)

**Question:** Did we retrieve all relevant chunks?

```python
from ragas.metrics import IDBasedContextRecall

metric = IDBasedContextRecall()

# Compares
retrieved_ids = ["B0BRJS644Z", "B099N9F3FP"]
reference_ids = ["B0BRJS644Z", "B099N9F3FP"]

# Score: 2 found / 2 needed = 1.0
```

**Interpretation:**
- **1.0** = All reference chunks retrieved
- **0.5** = Missing half the relevant chunks
- **0.0** = No relevant chunks retrieved

**Trade-off:** High recall often means lower precision (retrieving more = more noise)

#### Metric 3: Faithfulness (Hallucination Check)

**Question:** Is the answer grounded in the retrieved context?

```python
from ragas.metrics import Faithfulness

metric = Faithfulness(llm=ragas_llm)

# LLM analyzes
answer = "The Marame fan has 6 cooling modes and costs $15"
context = "Marame 120mm USB Fan with speed controller. Price: $14.99"

# LLM checks each claim:
# ✅ "has speed control" → verified in context
# ❌ "6 cooling modes" → NOT in context
# ✅ "costs $15" → approximately correct

# Score: 2 verified / 3 claims = 0.67
```

**Interpretation:**
- **1.0** = All claims verified in context (no hallucination)
- **0.5** = Half the answer is hallucinated
- **0.0** = Complete hallucination

#### Metric 4: Response Relevancy (Answer Quality)

**Question:** Does the answer actually address the question?

```python
from ragas.metrics import ResponseRelevancy

metric = ResponseRelevancy(llm=ragas_llm, embeddings=ragas_embeddings)

# Evaluates
question = "Do you have Bluetooth speakers?"
answer = "Yes, we have the Raymate Bluetooth Speaker with HiFi sound..."

# Uses embeddings + LLM to score relevance
# Score: 0.85 (highly relevant)
```

**Interpretation:**
- **1.0** = Perfect answer to the question
- **0.5** = Partially addresses question
- **0.0** = Irrelevant answer

### Evaluation Sample Structure

```python
from ragas.dataset_schema import SingleTurnSample

sample = SingleTurnSample(
    user_input="Do you have USB fans?",
    response="Yes, we have the Marame 120mm USB fan...",
    retrieved_contexts=[
        "Marame 120mm USB Powered Fan...",
        "AICHESON Laptop Cooler..."
    ],
    retrieved_context_ids=["B0BRJS644Z", "B099N9F3FP"],
    reference_context_ids=["B0BRJS644Z"]  # From eval dataset
)

# Evaluate
precision = await ragas_context_precision_id_based(sample)
recall = await ragas_context_recall_id_based(sample)
faithfulness = await ragas_faithfulness(sample)
relevancy = await ragas_response_relevancy(sample)
```

### Running Evaluations at Scale

```python
# Evaluate entire dataset
results = ls_client.evaluate(
    lambda x: rag_pipeline(x["question"], qdrant_client),
    data="rag-evaluation-dataset",  # LangSmith dataset
    evaluators=[
        ragas_context_precision_id_based,
        ragas_context_recall_id_based,
        ragas_faithfulness,
        ragas_response_relevancy
    ],
    experiment_prefix="retriever"
)
```

**View Results:** LangSmith Experiments dashboard

**Files:** `notebooks/week_1/06-RAG-Evals.ipynb`

---

## 7. Production Deployment

### System Architecture

```
┌─────────────┐         ┌─────────────┐         ┌─────────────┐
│  Streamlit  │ ──HTTP─→│   FastAPI   │ ──TCP──→│   Qdrant    │
│     UI      │         │     API     │         │   Vector    │
│  (Port 8501)│ ←──JSON─│  (Port 8000)│         │     DB      │
└─────────────┘         └─────────────┘         └─────────────┘
                              │
                              ↓
                        ┌──────────┐
                        │  OpenAI  │
                        │    API   │
                        └──────────┘
                              │
                              ↓
                        ┌──────────┐
                        │LangSmith │
                        │ Tracing  │
                        └──────────┘
```

### Component 1: FastAPI Backend

**File:** `apps/api/src/api/app.py`

```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Allow Streamlit to call API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# Health check
@app.get("/")
def read_root():
    return {"status": "healthy"}
```

**File:** `apps/api/src/api/api/endpoints.py`

```python
from fastapi import APIRouter
from .models import RAGRequest, RAGResponse
from ..agents.retrieval_generation import rag_pipeline

router = APIRouter()

@router.post("/rag", response_model=RAGResponse)
def rag_endpoint(request: RAGRequest):
    answer = rag_pipeline(request.query, qdrant_client)
    return RAGResponse(answer=answer)
```

**File:** `apps/api/src/api/api/models.py`

```python
from pydantic import BaseModel

class RAGRequest(BaseModel):
    query: str

class RAGResponse(BaseModel):
    answer: str
```

### Component 2: RAG Pipeline (Production)

**File:** `apps/api/src/api/agents/retrieval_generation.py`

```python
# Same code as Notebook 4, with all @traceable decorators
# This ensures production observability

@traceable(name="rag_pipeline")
def rag_pipeline(question: str, qdrant_client, top_k: int = 5) -> str:
    retrieved_context = retrieve_data(question, k=top_k)
    preprocessed_context = process_context(retrieved_context)
    prompt = build_prompt(preprocessed_context, question)
    answer = generate_answer(prompt)
    return answer
```

### Component 3: Streamlit UI

**File:** `apps/chatbot_ui/src/chatbot_ui/app.py`

```python
import streamlit as st
import requests

st.title("Product Recommendation Chatbot")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
if prompt := st.chat_input("Ask about products..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    # Call API
    try:
        response = requests.post(
            "http://api:8000/rag",
            json={"query": prompt}
        )
        answer = response.json()["answer"]

        # Add assistant response
        st.session_state.messages.append({"role": "assistant", "content": answer})

        with st.chat_message("assistant"):
            st.markdown(answer)

    except Exception as e:
        st.error(f"Error: {e}")
```

### Component 4: Docker Compose

**File:** `docker-compose.yml`

```yaml
services:
  streamlit-app:
    build:
      context: ./apps/chatbot_ui
    ports:
      - "8501:8501"
    volumes:
      - ./apps/chatbot_ui:/app
    environment:
      - API_URL=http://api:8000
    depends_on:
      - api

  api:
    build:
      context: ./apps/api
    ports:
      - "8000:8000"
    volumes:
      - ./apps/api:/app
      - ./.env:/app/.env
    environment:
      - QDRANT_URL=http://qdrant:6333
    depends_on:
      - qdrant

  qdrant:
    image: qdrant/qdrant:latest
    ports:
      - "6333:6333"
      - "6334:6334"
    volumes:
      - ./qdrant_data:/qdrant/storage
```

### Running the System

```bash
# Start all services
make run-docker-compose
# OR
docker-compose up

# Access services
# UI: http://localhost:8501
# API Docs: http://localhost:8000/docs
# Qdrant Dashboard: http://localhost:6333/dashboard
```

### Component 5: Evaluation Script

**File:** `apps/api/evals/eval_retriever.py`

```python
from langsmith import Client
from ragas.metrics import (
    IDBasedContextPrecision,
    IDBasedContextRecall,
    Faithfulness,
    ResponseRelevancy
)

ls_client = Client()

# Run evaluation
results = ls_client.evaluate(
    lambda x: rag_pipeline(x["question"], qdrant_client),
    data="rag-evaluation-dataset",
    evaluators=[
        ragas_context_precision_id_based,
        ragas_context_recall_id_based,
        ragas_faithfulness,
        ragas_response_relevancy
    ],
    experiment_prefix="retriever",
    max_concurrency=4
)

print(results)
```

**Run:** `python apps/api/evals/eval_retriever.py`

---

## Key Takeaways

### Conceptual Understanding

1. **RAG = Search + Generation**
   - Retrieval grounds the LLM in actual data
   - Reduces hallucinations
   - Enables up-to-date information

2. **Vector Embeddings Capture Meaning**
   - Semantic similarity vs keyword matching
   - Same concept, different words → similar vectors
   - Example: "USB fan" matches "USB-powered cooling"

3. **Observability is Critical**
   - Track token usage (costs)
   - Debug retrieval quality
   - Identify bottlenecks

4. **Evaluation Drives Quality**
   - Metrics reveal weaknesses
   - Compare approaches objectively
   - Catch regressions before production

### Technical Patterns

1. **Text Preprocessing**
   ```python
   # Concatenate relevant fields
   text = f"{title} {' '.join(features)}"
   ```

2. **Retrieval Pattern**
   ```python
   # Embed → Search → Return top-k
   embedding = get_embedding(query)
   results = vector_db.query(embedding, limit=k)
   ```

3. **Context Formatting**
   ```python
   # Structure for LLM consumption
   context = "\n".join([f"- {item}" for item in results])
   ```

4. **Prompt Engineering**
   ```python
   # System instructions + Context + Question
   prompt = f"{instructions}\n\n{context}\n\nQ: {question}"
   ```

5. **Tracing Pattern**
   ```python
   @traceable(name="...", run_type="...")
   def function():
       # Automatic logging to LangSmith
   ```

### Common Pitfalls

1. **Don't Skip Data Exploration**
   - Understanding your data shapes preprocessing choices
   - Rating distributions, price ranges matter

2. **Match Embedding Dimensions**
   - Vector DB size must match embedding model output
   - `text-embedding-3-small` = 1536 dimensions

3. **Top-k is a Trade-off**
   - Too low: Miss relevant info (low recall)
   - Too high: Add noise (low precision)
   - Start with k=5, tune based on metrics

4. **Prompt Formatting Matters**
   - Clear instructions reduce hallucinations
   - Structured context helps LLM parse info

5. **Always Track Token Usage**
   - Costs add up in production
   - Optimize prompt length and model choice

### Production Checklist

- ✅ Environment variables for API keys
- ✅ Error handling in API endpoints
- ✅ CORS configuration for UI
- ✅ Persistent vector database storage
- ✅ Logging and tracing enabled
- ✅ Evaluation metrics in CI/CD
- ✅ Docker for reproducible deployment

---

## Quick Commands Reference

### Environment Setup (uv)

```bash
# Create virtual environment
uv venv

# Activate environment
source .venv/bin/activate  # macOS/Linux
.venv\Scripts\activate     # Windows

# Install all dependencies (base + dev)
uv sync --group dev

# Install workspace dependencies
uv sync --workspace --group dev

# Add new package to dev group
uv add --group dev package-name

# Update all packages
uv sync --upgrade
```

### Development

```bash
# Run Jupyter notebooks
uv run jupyter notebook notebooks/week_1/

# Or with activated venv
jupyter notebook notebooks/week_1/

# Start Qdrant locally
docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant:latest

# Run Python script
uv run python script.py

# Check installed packages
uv pip list
```

### Production

```bash
# Build and run all services
docker-compose up --build

# Run evaluations
uv run python apps/api/evals/eval_retriever.py

# View logs
docker-compose logs -f api

# Stop services
docker-compose down
```

### Data Pipeline

```bash
# 1. Filter dataset
# Run: notebooks/week_1/01-explore-amazon-dataset.ipynb

# 2. Embed and index
# Run: notebooks/week_1/02-RAG-preprocessing-items.ipynb

# 3. Generate eval dataset
# Run: notebooks/week_1/05-RAG-Eval-Dataset.ipynb

# 4. Run evaluations
# Run: notebooks/week_1/06-RAG-Evals.ipynb
```

### LangSmith

```bash
# Set up environment variables in .env
LANGSMITH_API_KEY=lsv2_pt_...
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=ai-engineering-bootcamp-week1

# View traces at: https://smith.langchain.com/
# View datasets at: https://smith.langchain.com/datasets
# View experiments at: https://smith.langchain.com/experiments
```

---

## Technologies Used

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **LLM** | OpenAI GPT-5.4-nano | Answer generation |
| **Embeddings** | text-embedding-3-small | Semantic vectors (1536-dim) |
| **Vector DB** | Qdrant | Similarity search |
| **Observability** | LangSmith | Tracing, datasets, and evaluation experiments |
| **Eval Framework** | RAGAS | RAG-specific metrics (precision, recall, faithfulness, relevancy) |
| **API** | FastAPI | Backend service |
| **UI** | Streamlit | Chat interface |
| **Deployment** | Docker Compose | Multi-container orchestration |
| **Package Manager** | uv | Fast Python package/environment management |
| **Data Processing** | Pandas | Data filtering and preprocessing |
| **Language** | Python 3.12+ | Required for modern async features |

---

## Next Steps (Week 2 Preview)

Based on upstream branches, Week 2 likely covers:

- **Structured Outputs:** Pydantic models for LLM responses
- **Hybrid Search:** Combining vector + keyword search
- **Reranking:** Re-scoring retrieved results
- **Prompt Management:** Version control for prompts
- **Larger Eval Datasets:** Scaling evaluation

---

## Additional Resources

### Core Technologies
- **LangSmith Docs:** https://docs.smith.langchain.com/
- **LangSmith Platform:** https://smith.langchain.com/ (sign up for free account)
- **RAGAS Docs:** https://docs.ragas.io/
- **Qdrant Docs:** https://qdrant.tech/documentation/
- **uv Docs:** https://docs.astral.sh/uv/
- **FastAPI Docs:** https://fastapi.tiangolo.com/
- **Streamlit Docs:** https://docs.streamlit.io/
- **OpenAI API Docs:** https://platform.openai.com/docs/

### Learning Resources
- **RAG Papers:** https://arxiv.org/abs/2005.11401 (Original RAG paper)
- **Vector Search:** https://www.pinecone.io/learn/vector-database/
- **Prompt Engineering:** https://www.promptingguide.ai/

### Community
- **LangChain Discord:** https://discord.gg/langchain
- **Qdrant Community:** https://discord.gg/qdrant

---

**Study Tip:** Run each notebook sequentially, understanding each step before moving to the next. The progression is intentional: setup → data → embeddings → RAG → observability → evaluation → production.
