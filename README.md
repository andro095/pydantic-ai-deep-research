# Pydantic AI Deep Research Agent with Gradio UI

A modern, visually polished web interface wrapping an autonomous **Deep Research Agent** built using the **Pydantic AI** framework, powered by Google's **Gemini 3.1 Flash** (or configurable) model.

The agent takes either a free-text research query or a stock ticker (e.g. `NVDA`) and produces a structured, detailed research report with inline citations, evidence tables, risk analysis, and a "what to watch next" checklist.

---

## Features
- **Intent & Entity Detection:** Automatically detects if the input is a stock ticker symbol. If true, resolves it to the full company name and sector keywords (e.g., `NVDA` -> `NVIDIA Corporation`).
- **Resilient Web Search:** Custom scrapers targeting Google Search, with an automatic, silent fallback to DuckDuckGo HTML search if Google rates-limits or blocks requests.
- **Parallel Deep Dives (Async):** Generates 3-4 specific, non-overlapping research angles (e.g. SWOT, financial results, competitive environment), then scrapes and fetches target webpage contents concurrently using `asyncio.gather`.
- **Citing & Claim Tracking:** Uses Pydantic AI to extract key statistics, facts, and numbers, mapping each claim back to its source URL and page title.
- **Report Synthesis:** Compiles all findings into a structured report following expert analyst constraints (including tables, risk matrices, and checklist).
- **Streaming UI Progress Logs:** Real-time state updates (e.g., intent detection, research angle generation, parallel scraping status) streamed in the Gradio chat window before presenting the final report.

---

## Project Structure
```
├── .env                  # Configuration keys (API key and model selection)
├── .env.example          # Template for local environment setup
├── requirements.txt      # Python dependencies (Pydantic AI, Gradio, python-dotenv, BeautifulSoup4)
├── agent_module.py       # Scrapers, parallel fetchers, intent detectors, and orchestration
├── app.py                # Gradio dark glassmorphic web UI with async streaming predict function
└── README.md             # This guide
```

---

## Setup & Running Instructions

### 1. Clone or Open the Workspace
Ensure you are in the directory containing this project:
```bash
cd /Users/andro095/Documents/dev/Courses/agents_course/pydantic
```

### 2. Create and Activate a Virtual Environment
Initialize a fresh Python virtual environment (`venv`):
```bash
# Create the venv
python3 -m venv venv

# Activate it (macOS/Linux)
source venv/bin/activate
```

### 3. Install Dependencies
Install all required packages from `requirements.txt`:
```bash
pip install -r requirements.txt
```

### 4. Setup Your Environment Variables
Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```
Open `.env` and add your Google Gemini API Key:
```env
GEMINI_API_KEY=your_actual_api_key_here
GEMINI_MODEL=google:gemini-3.1-flash-lite-preview
```
*Note: You can get a Gemini API Key for free from [Google AI Studio](https://aistudio.google.com/).*

### 5. Launch the Web Application
Start the Gradio frontend:
```bash
python app.py
```

Open your browser and navigate to:
**`http://localhost:7860`**

---

## How It Works

1. **Orchestrator (`agent_module.py`):**
   - **Intent detection:** Calls a prompt model to classify whether the query is a ticker or text.
   - **Discovery:** Scrapes initial search results for query context.
   - **Angles generation:** Prompts Gemini to produce 3-4 distinct research angles.
   - **Parallel deep dives:** Concurrently runs searches, downloads webpage text content via `httpx` + `BeautifulSoup`, and calls Gemini in parallel to extract cited facts.
   - **Synthesis:** Passes all citations and facts to a final agent run to construct a formal structured report.

2. **Frontend UI (`app.py`):**
   The Gradio app uses an `async def predict(...)` generator. Each step yields progress messages so the user gets instant, real-time logging feedback on the status of the web searches and scrapes, culminating in the rendering of the final Markdown report.
