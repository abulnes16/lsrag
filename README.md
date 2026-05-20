# L-SRAG (Light Search Retrieval-Augmented Generation)

L-SRAG is a full-stack, containerized application designed to evaluate, compare, and experiment with different Retrieval-Augmented Generation (RAG) paradigms. It provides a side-by-side comparison interface between a baseline **NaiveRAG** approach and an advanced **LightRAG** system. The platform allows users to not only chat with the knowledge base but also evaluate the quality of the generated answers against ground-truth references using industry-standard semantic and ranking metrics.

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

## 📄 License
This project was built for evaluating Search Augmented Generation techniques. All rights reserved.
