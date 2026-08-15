# Finance RAG

A retrieval-augmented generation (RAG) system for answering questions over SEC filings (10-Ks) for Microsoft, NVIDIA, and JPMorgan Chase, using hybrid retrieval and Gemini for generation.

## Project structure

finance-rag/
├── data/
│ └── filings/ # raw + extracted SEC filing text (HTML not tracked in git)
├── evaluation/
│ ├── test_questions.json
│ ├── evaluate.py # runs full eval suite
│ ├── debug_retrieval.py # inspect retrieval for a single question
│ ├── debug_bm25.py # compare BM25 vs embedding ranking
│ ├── find_chunk.py # search raw chunks for a specific string
│ └── results.json # latest eval output
├── src/
│ ├── ingest.py # HTML -> clean text extraction
│ ├── chunk_and_embed.py # text -> chunks (data/chunks.json)
│ ├── embed_and_store.py # chunks -> embeddings -> Chroma vector store
│ └── query.py # retrieval (hybrid BM25 + embeddings) + generation
└── chroma_db/ # persistent vector store (not tracked in git)


## How it works

1. **Ingest** (`src/ingest.py`) — parses raw SEC filing HTML in `data/filings/`, extracts tables and text into `*_extracted.txt` files.
2. **Chunk** (`src/chunk_and_embed.py`) — splits filings into ~500-character overlapping chunks (LangChain's recursive text splitter), saves to `data/chunks.json`.
3. **Embed & store** (`src/embed_and_store.py`) — embeds chunks with `all-MiniLM-L6-v2` (sentence-transformers) and stores them in a persistent Chroma vector database.
4. **Query** (`src/query.py`) — retrieves relevant chunks and generates answers using Gemini.

## Retrieval: hybrid search

Initial retrieval used embeddings only, which struggled with two failure modes on financial filings:
- **Near-duplicate boilerplate across years** — e.g. "For fiscal year 2024, sales to one customer represented 13%..." vs. the FY2026 version with 22% — nearly identical phrasing confused pure embedding similarity.
- **Table vs. prose mismatch** — dense financial tables (e.g. `Net income | 133,749 | 101,832 | 31%`) embed poorly against natural-language questions.

Fixed by combining BM25 keyword search with embedding search via **Reciprocal Rank Fusion**, which resolved the near-duplicate-year case. The table-vocabulary-mismatch case remains a known limitation (documented below).

## Evaluation

18 test questions across easy/medium/hard difficulty, auto-scored by keyword matching against expected answers.

**Latest results:** [update once your final run completes]

Run: `python evaluation/evaluate.py`

Debug tools:
- `python evaluation/debug_retrieval.py` — trace which chunks get retrieved for a specific question
- `python evaluation/debug_bm25.py` — compare BM25-only vs embedding-only ranking for a chunk
- `python evaluation/find_chunk.py` — grep raw chunks for a specific string/figure

## Known limitations

- One question (Microsoft net income comparison) fails retrieval - the answer lives in a densely formatted summary table whose column headers ("2026", "2025") don't lexically match natural question phrasing ("fiscal year 2026"). Diagnosed via chunk-level debugging; the fix would be enriching table chunks with synthetic natural-language context at indexing time.
- Free-tier Gemini API has strict per-minute and per-day rate limits; `evaluate.py` includes retry logic and response caching (`evaluation/answer_cache.json`) to handle this.

## Setup

```bash
pip install -r requirements.txt
```

Create a `.env` file in the project root with:

GEMINI_API_KEY=your_key_here


Then run the pipeline in order:
```bash
python src/ingest.py
python src/chunk_and_embed.py
python src/embed_and_store.py
python src/query.py            # interactive Q&A
python evaluation/evaluate.py  # run test suite
```