# LSRAG (Light Sovereign Retrieval-Augmented Generation)

LSRAG is a full-stack, containerized application designed to evaluate, compare, and experiment with a RAG system using SLM locally to preserve the sovereignity of the data by keeping it in the user local computer. It functions as a experimental prototype of RAG system using Graph and Vectorial retrieval  using SLM. It provides a side-by-side comparison interface between a baseline **NaiveRAG** approach and an advanced **LightRAG** system. The platform allows users to not only chat with the knowledge base but also evaluate the quality of the generated answers against ground-truth references using industry-standard semantic and ranking metrics.

## 🌟 Key Features

- **Split Chat Interface**: Chat with both NaiveRAG and LightRAG side-by-side simultaneously.
- **Multiple LightRAG Modes**: Test LightRAG using `Mix` (default), `Local`, `Global`, and `Hybrid` execution modes.
- **Live Evaluation Dashboard**: Automatically evaluate RAG answers against a Ground Truth Reference using robust metrics:
  - **Faithfulness**: Measures factual consistency of the generated answer against the retrieved contexts.
  - **Context Recall**: Measures alignment between the expected reference and retrieved contexts.
  - **Answer Relevancy**: Measures the semantic relevance of the generated response to the user's query.
  - **NDCG (Ranking)**: Measures the ranking quality of retrieved context chunks using cosine similarity.
- **Automated Data Provisioning**: Automatically ingests datasets (e.g., msmarco-qa, hotpot-qa) into the local file system for querying.

## 🏗 Architecture

The project is structured into three main Dockerized components:

1. **Frontend (`/frontend`)**: 
   - Built with **Next.js 15 (App Router)**, **React 19**, and **TypeScript**.
   - Highly decoupled architecture using custom React Hooks (`useChat`, `useExperiment`) and shared UI components.
   - Styled beautifully with **Tailwind CSS**, featuring dark mode, glassmorphic layouts, and responsive design.

2. **Backend (`/backend`)**:
   - High-performance API built with **FastAPI** and **Python**.
   - Integrates with local/hosted LLMs via **Ollama** (`host.docker.internal`).
   - Uses **Ragas** and **scikit-learn** to generate live evaluation metrics on LLM responses and embedded context chunks.

3. **Data Provider (`/data-provider`)**:
   - A short-lived initialization container that fetches and sets up the embedding/vector data in the shared `./data` volume.

## Design Decisions

This section provides the theoretical and practical justifications for the primary hyperparameters and architectural decisions made in the LSRAG system.

### 1. RAG Hyperparameters

| Parameter | NaiveRAG Value | LightRAG Value | Architectural Rationale |
| :--- | :--- | :--- | :--- |
| **Chunk Size (`chunk_size`)** | `200` words | `300` tokens | **Fair Baseline Comparison & Embedding Limits:** LightRAG splits text by *tokens* (using tiktoken), whereas NaiveRAG splits text by *words* (whitespace-based). Because 1 English word $\approx$ 1.35–1.4 tokens, a 200-word Naive chunk is physically equivalent to about 270–280 tokens. This makes the text blocks compared between systems almost identical. Additionally, the local embedding model `mxbai-embed-large` has a hard context limit of **512 tokens**; keeping chunks under 300 tokens guarantees no text is lost to truncation during embedding. |
| **Chunk Overlap (`chunk_overlap`)** | `50` words | `50` tokens | **Semantic Continuity:** A 15–25% overlap acts as a safety boundary. It prevents semantic fragmentation by ensuring that entities or core facts that are split across chunk borders are preserved and readable in both adjacent contexts. |
| **Top-K Retrieval (`top_k`)** | `5` | `5` | **Focused Context & Evaluation Safety:** We modified the LightRAG retrieval parameters (`top_k` and `chunk_top_k`) to dynamically match NaiveRAG's query limit (`num=5`). This ensures both RAG systems generate their final answers and evaluate metrics on the same retrieval depth. It also limits the prompt token count passed to the LLM and the evaluation judge, preventing API timeouts and cost inflation. |
| **Embedding Dimension (`embed_dim`)** | `1024` | `1024` | **Model Alignment:** This parameter is strictly determined by the embedding model architecture. `mxbai-embed-large` produces 1024-dimensional vectors, so the vector database storage schema (`NanoVectorDB`) must match this value to perform accurate cosine similarity queries. |

### 2. Model Selection Strategy

*   **Data Sovereignty & Local SLM (`phi3:mini`)**:
    *   *Decision:* Local execution of the primary LLM is a core requirement to keep private datasets (like corporate GitLab workflows) secure on local hardware.
    *   *Why Phi-3 Mini?* At 3.8B parameters, it has a tiny footprint and runs with sub-second latency on consumer hardware, but its training on heavily filtered datasets gives it reasoning and structured JSON output capabilities that rival standard 7B–8B parameter models.
*   **Local Embedding (`mxbai-embed-large`)**:
    *   *Decision:* Runs locally via Ollama to compute vectors. It is a state-of-the-art open-source search embedding model.
*   **Evaluation Judge (`gpt-4o-mini`)**:
    *   *Decision:* Why use a cloud model for evaluation? Semantic metrics (Faithfulness, Recall, Relevancy) require a high-capacity, deterministic judge model. Small local models (SLMs) lack the reasoning capabilities and JSON-formatting compliance required to run RAGAS pipelines reliably, leading to parsing failures.

### 3. Handling Context & Token Limit Overflows

During multi-hop evaluation, the judge LLM analyzes large context blocks and outputs complex reasoning paths. To guarantee successful metric calculations, we implemented:
*   **Output Token Expansion (`max_tokens=8192`)**: Increased Ragas's completion budget from the default `3072` tokens to `8192` to prevent truncated/corrupted JSON responses.
*   **Context Truncation (`6000` characters)**: Added a sentence-aware truncation helper in the metric calculator. If the retrieved context is excessively large, it caps the context size at ~1,000 words. This retains all necessary facts to calculate faithfulness/recall while keeping API prompt sizes concise and highly performant.

## 🚀 Getting Started

### Prerequisites
- [Docker](https://www.docker.com/) and Docker Compose installed.
- [Ollama](https://ollama.com/) running locally (if testing with local models).
- An OpenAI API Key (for Ragas metrics calculation).

### Installation & Setup

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd lsrag
   ```

2. **Configure Environment Variables**:
   Create a `.env` file inside the `backend/` directory and populate it with your OpenAI API Key:
   ```bash
   OPENAI_API_KEY=your_openai_api_key_here
   ```

3. **Start the Application**:
   Use Docker Compose to build and start the entire stack:
   ```bash
   docker compose up --build
   ```

   *Note: The `data-provider` container will execute first to seed the `./data` directory. Wait for the `frontend` and `backend` containers to report that they are fully running.*

4. **Access the Application**:
   - **Frontend UI**: Open your browser and navigate to [http://localhost:3000](http://localhost:3000)
   - **Backend API Docs (Swagger)**: Navigate to [http://localhost:8000/docs](http://localhost:8000/docs)

## 📖 Usage

### Standard Chatting
1. Go to the root path (`/`) to interact with the standard chat module.
2. Ask questions against the ingested knowledge base.

### Experimentation & Evaluation
1. Navigate to `/experiment` to open the evaluation dashboard.
2. Enter your **Question**.
3. Select your preferred **LSRAG Mode**.
4. To run an evaluation, you **must** enter a **Ground Truth Reference** (the golden answer you expect the system to hit).
5. Click **Run Evaluation** to query both NaiveRAG and LightRAG. Once they finish responding, the Live Scores Table will populate with the semantic similarity and ranking metric outcomes.

## 🛠 API Endpoints

The backend exposes the following primary POST endpoints:

### `POST /chat`
Generates a response from the designated RAG system without evaluating metrics.
**Payload:**
```json
{
  "query": "In which year the cold war was held?",
  "rag_type": "lightrag", // or "naiverag"
  "lightrag_mode": "mix"
}
```

### `POST /evaluate`
Queries both NaiveRAG and LightRAG simultaneously, extracts retrieved chunks, calculates Ragas & NDCG metrics against the reference, and returns the full telemetry.
**Payload:**
```json
{
  "query": "In which year the cold war was held?",
  "reference": "The Cold War was held between 1947 and 1991",
  "lightrag_mode": "mix"
}
```
## 🌍 Environment Switching

The application supports switching between two environments: **experiment** (standard public QA datasets) and **corporate** (GitLab Engineering Workflow dataset).

The working directories and datasets loaded by the backend services are controlled by the `LIGHTRAG_DIR` environment variable, configured via Docker Compose environment files:

| Environment | Env File | LightRAG Cache Dir | NaiveRAG Cache Dir | Ingested Dataset |
| :--- | :--- | :--- | :--- | :--- |
| **experiment** (Default) | `.env.experiment` | `lightrag_cache` | `naive_data` | `msmarco-qa` & `hotpotqa` |
| **corporate** | `.env.corporate` | `corporate_lightrag_cache` | `corporate_naive_data` | GitLab Engineering Workflow (`*.md`) |

### How to Switch and Start the Containers

1. **Start the Experiment Environment**:
   To run the default experiment configuration:
   ```bash
   docker compose --env-file .env.experiment up --build
   ```
   *(Or simply run `docker compose up --build` since the default `.env` file points to the experiment configuration).*

2. **Start the Corporate Environment**:
   To run the corporate configuration:
   ```bash
   docker compose --env-file .env.corporate up --build
   ```
   *In this environment, the backend automatically reads and ingests the GitLab Markdown documents from `./corporate_dataset/gitlab_engineering_workflow_dataset` (mounted into the container).*

## 📄 License
This project was built for evaluating Search Augmented Generation techniques. All rights reserved.
