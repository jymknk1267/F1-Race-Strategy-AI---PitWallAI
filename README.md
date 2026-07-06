# PitWallAI 🏎️

A domain-specific Retrieval-Augmented Generation (RAG) system that answers F1 race strategy questions using a curated circuit knowledge base. Ask it anything about tyre compounds, pit windows, undercuts, or safety car scenarios — and get a response grounded in real circuit data, delivered in the voice of a race engineer.

---

## How It Works

1. A curated Markdown knowledge base covers circuit characteristics, tyre behaviour, historical strategy patterns, weather tendencies, safety car probabilities, and undercut/overcut effectiveness across multiple F1 venues
2. The knowledge base is chunked and embedded into a ChromaDB vector store using OpenAI's `text-embedding-3-small` model
3. User queries in natural language are semantically matched against the vector store
4. Retrieved context is passed to GPT-4.1 Nano alongside a race engineer persona prompt, producing circuit-grounded strategy recommendations

---

## Project Structure

```
├── data/
│   └── batches/          # Markdown knowledge base files
├── chroma/               # ChromaDB vector store (auto-generated)
├── createdb.py           # Ingests and embeds knowledge base into ChromaDB
├── query.py              # Interactive query interface
├── .env                  # API key (not committed)
├── .gitignore
└── requirements.txt
```

---

## Setup

**1. Clone the repository and create a virtual environment**
```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Add your OpenAI API key to `.env`**
```
OPENAI_API_KEY=your-key-here
```

**4. Build the vector store**
```bash
python createdb.py
```

**5. Run the strategy assistant**
```bash
python query.py
```

---

## Example Queries

```
I'm P4 at Monza on lap 30 on Medium tyres with heavy degradation. Box now or stay out?

Safety car just deployed at Baku on lap 22. I'm on Hards with 29 laps to go. What's the call?

Abu Dhabi, lap 15, P7 on Mediums. Undercut or overcut the car ahead?
```

---

## Tech Stack

| Tool | Purpose |
|------|---------|
| Python | Core language |
| LangChain | Document loading, chunking, prompt orchestration |
| ChromaDB | Vector store and similarity search |
| OpenAI Embeddings (`text-embedding-3-small`) | Chunk vectorisation |
| GPT-4.1 Nano | Strategy response generation |
| python-dotenv | API key management |

---

## Circuits Currently Covered

- Australia
- Bahrain
- Canada
- Monaco
- Barcelona Catalunya, Spain
- Red Bull Ring, Austria
- Silverstone, Great Britain
- Belgium
- Monza, Italy
- Yas Marina Circuit, Abu Dhabi
- Baku, Azerbaijan

*Additional circuits can be added by extending the Markdown knowledge base and rerunning `createdb.py`.*
