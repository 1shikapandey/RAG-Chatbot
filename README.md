# Knowledge Assistant & Evaluation Dashboard

An AI-powered knowledge assistant and evaluation system. 

This system uses a advanced hybrid RAG (Retrieval-Augmented Generation) pipeline to answer user questions using the provided community documents, handles multi-document reasoning, avoids hallucinations by correctly identifying unanswerable questions, and contains an automated evaluation suite to assess performance using LLM-as-a-judge metrics.

---

## 🌟 Key Features

1. **Hybrid Retrieval**: Combines keyword search (**BM25**) with semantic dense search (**Gemini text-embedding-004**) combined using **Reciprocal Rank Fusion (RRF)**.
2. **Local Caching**: Embeddings of document chunks are cached locally in `.embeddings_cache.json` after the first generation. This saves API quota and makes document loading instantaneous.
3. **Anti-Hallucination Grounding**: Prompt engineering binds response output strictly to retrieved context. Unanswerable questions are caught and answered with a standard refusal message rather than made-up facts.
4. **LLM-as-a-judge Evaluation Suite**: Automated assessment of 15 diverse test cases measuring:
   - **Context Precision**: relevance of retrieved segments.
   - **Context Recall**: whether the reference answer can be derived from retrieved segments.
   - **Answer Relevancy**: whether the assistant's answer addresses the question directly.
   - **Faithfulness**: whether the generated response is strictly grounded in retrieved facts.
5. **Vibrant Glassmorphic Web UI**: A modern dashboard providing:
   - Interactive chat window with grounding sources.
   - **Workspace Citation Link Highlights**: Clicking a chat citation automatically opens that source document in the inspector and scrolls/highlights the exact line range!
   - Chart.js visualization of evaluation metrics.
   - Interactive review modal for each evaluation case showing question, reference answer, generated answer, and detailed judge explanations.

---

## 🛠️ Installation & Setup

Ensure you have **Python 3.11+** installed. Using `uv` is recommended for high-speed package management, though standard `pip` works too.

### 1. Clone & Set working directory
Verify that the project structure contains the knowledge base documents in the `About/` folder:
```
knowledge_base/ (or About/)
├── about.txt
├── code_of_conduct.txt
├── domains.txt
├── hackathons.txt
├── leadership.txt
└── membership.txt
```

### 2. Configure Environment (API Key)
The system uses Google Gemini for answer generation, dense embeddings, and the evaluation judges. 
Create a `.env` file in the project root and add your API key:
```env
GEMINI_API_KEY=your_google_gemini_api_key_here
```
*Note: You can also paste and save the API key directly from the Web interface at runtime! If no key is set, the application automatically runs in offline BM25 mock mode.*

### 3. Create Virtual Environment & Install Dependencies
Using `uv`:
```powershell
uv venv .venv
uv pip install -r requirements.txt
```
Using standard `pip`:
```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

---

## 🚀 Running the System

### 1. Launch the Web Interface
Start the Flask development server:
```powershell
.venv\Scripts\python app.py
```
Open your browser and navigate to **`http://127.0.0.1:5000`** to access the dashboard.

### 2. Run the Evaluation Suite via CLI
If you wish to execute the evaluation set directly from the terminal:
```powershell
.venv\Scripts\python evaluator.py
```
This will run the 15 test cases in `eval_set.json` and output a summary report in the console, saving detailed metrics to `eval_results.json`.

---

## 📘 Technical Approach & Major Decisions

### 1. Text Parsing & Paragraph-level Chunking
Instead of standard arbitrary character-length splitting (which ruins sentences and structures), documents are parsed by empty line breaks (paragraphs). This retains clean factual statements. The chunker also logs the precise line number range of each chunk (`start_line` and `end_line`), allowing the frontend document viewer to highlight lines that matched.

### 2. BM25 + Gemini Embeddings Hybrid Retrieval
Lexical search (BM25) is robust for exact matching of community acronyms, domain names, or names (e.g. "VIT Pune", "Web3", "dApps"). Dense retrieval (Gemini embeddings) is robust for semantic matching of conceptual descriptions (e.g. conflict resolution or project-building goals). 
Reciprocal Rank Fusion (RRF, with constant $k=60$) is used to combine both lists, scoring chunks based on their ranked position in both search models.

### 3. Zero-Hallucination Prompt Grounding
The prompt template includes strict instructions binding the model to the context chunks. If the question cannot be answered from the chunks, the model must output exactly:
`I am sorry, but the provided knowledge base does not contain information to answer this question.`
This prevents hallucinated answers for out-of-domain questions.

### 4. Custom Evaluation Judges
Instead of relying on heavy packages (RAGAS/DeepEval) which frequently break during installation on local environments, we implemented custom LLM-as-a-judge prompts using Gemini's structured JSON output mode (`response_mime_type="application/json"`). This evaluates context precision, recall, answer relevancy, and faithfulness programmatically and returns the exact reasoning from the LLM judge.
