import os
import json
import asyncio
import httpx
import urllib.parse
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from pydantic_ai import Agent
from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, UserPromptPart
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.providers.google import GoogleProvider
from typing import Any, AsyncGenerator

# Load environment variables from .env
load_dotenv()

# Configured model name - defaults to google:gemini-3.1-flash-lite-preview or google:gemini-3.5-flash
RAW_MODEL_NAME = os.getenv("GEMINI_MODEL", "google:gemini-3.1-flash-lite-preview")
MODEL_NAME = RAW_MODEL_NAME.replace("google:", "")

# Define system prompt for short answers (kept for compatibility)
SYSTEM_PROMPT = (
    "You are a helpful, extremely concise assistant. "
    "Your answers must be very short, brief, and directly to the point. "
    "Use bullet points or single sentences wherever possible. "
    "Do not write long explanations unless explicitly requested by the user."
)

# Initialize a default agent (kept for compatibility)
agent = Agent(system_prompt=SYSTEM_PROMPT)

def get_agent_model(api_key: str | None = None) -> GoogleModel | str:
    """
    Resolves the model to use. If a custom API key is supplied (e.g. from the UI),
    it constructs a GoogleModel bound to a GoogleProvider with that key.
    Otherwise, it checks environment variables or falls back to the raw model string.
    """
    key_to_use = None
    if api_key and api_key.strip():
        key_to_use = api_key.strip()
    else:
        key_to_use = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    
    if key_to_use and key_to_use.strip():
        provider = GoogleProvider(api_key=key_to_use.strip())
        return GoogleModel(MODEL_NAME, provider=provider)
    
    return RAW_MODEL_NAME

async def run_agent_with_key_async(system_prompt: str, prompt: str, api_key: str | None = None) -> str:
    """
    Helper function to run a one-off agent session asynchronously.
    """
    temp_agent = Agent(system_prompt=system_prompt)
    model = get_agent_model(api_key)
    result = await temp_agent.run(prompt, model=model)
    return result.output

def _extract_text(val) -> str:
    if isinstance(val, str):
        return val
    if isinstance(val, dict):
        return val.get("text", "") or ""
    
    # Handle Gradio TextMessage / Pydantic models / objects with attributes
    if hasattr(val, "text"):
        return getattr(val, "text", "") or ""
    
    if hasattr(val, "model_dump"):
        try:
            d = val.model_dump()
            if isinstance(d, dict):
                return d.get("text", "") or ""
        except Exception:
            pass
            
    if hasattr(val, "dict"):
        try:
            d = val.dict()
            if isinstance(d, dict):
                return d.get("text", "") or ""
        except Exception:
            pass
            
    # Handle list of items
    if isinstance(val, (list, tuple)):
        parts = []
        for subval in val:
            extracted = _extract_text(subval)
            if extracted:
                parts.append(extracted)
        return " ".join(parts)
        
    return str(val)

def _extract_role_and_content(item) -> tuple[str | None, Any]:
    if isinstance(item, dict):
        return item.get("role"), item.get("content", "")
    
    role = getattr(item, "role", None)
    content = getattr(item, "content", "")
    return role, content

def chat_with_agent(message: str | dict, history: list, api_key: str | None = None) -> str:
    """
    Interacts with the Pydantic AI agent (kept for compatibility).
    """
    message = _extract_text(message)
    pydantic_history = []
    for item in history:
        role, content_val = _extract_role_and_content(item)
        if role:
            content = _extract_text(content_val)
            if role == "user":
                pydantic_history.append(ModelRequest(parts=[UserPromptPart(content=content)]))
            elif role == "assistant":
                pydantic_history.append(ModelResponse(parts=[TextPart(content=content)]))
        
    model = get_agent_model(api_key)
    try:
        result = agent.run_sync(message, message_history=pydantic_history, model=model)
        return result.output
    except Exception as e:
        return f"⚠️ Error: {str(e)}"

# =====================================================================
# DEEP RESEARCH AGENT CORE UTILITIES
# =====================================================================

async def google_search_scrape_async(query: str, num_results: int = 5) -> list[dict]:
    """
    Scrapes Google search results for a given query to extract Title, URL, and Snippet.
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    url = f"https://www.google.com/search?q={urllib.parse.quote(query)}"
    try:
        async with httpx.AsyncClient(timeout=10.0, follow_redirects=True) as client:
            response = await client.get(url, headers=headers)
        if response.status_code != 200:
            return []
        
        soup = BeautifulSoup(response.text, "html.parser")
        results = []
        for g in soup.select("div.g")[:num_results]:
            title_el = g.select_one("h3")
            link_el = g.select_one("a")
            snippet_el = (
                g.select_one("div[style*='-webkit-line-clamp']") 
                or g.select_one("span.aCOpbc") 
                or g.select_one(".VwiC3b")
            )
            
            if title_el and link_el:
                title = title_el.get_text()
                href = link_el.get("href")
                if href and href.startswith("/url?q="):
                    href = urllib.parse.parse_qs(urllib.parse.urlparse(href).query).get("q", [href])[0]
                elif href and (href.startswith("/") or "google.com" in href):
                    continue
                snippet = snippet_el.get_text() if snippet_el else ""
                results.append({
                    "title": title,
                    "url": href,
                    "snippet": snippet
                })
        return results
    except Exception as e:
        print(f"[ERROR] Google search scrape error: {e}")
        return []

async def ddg_search_scrape_async(query: str, num_results: int = 5) -> list[dict]:
    """
    Scrapes DuckDuckGo HTML search as a resilient fallback.
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    url = f"https://html.duckduckgo.com/html/?q={urllib.parse.quote(query)}"
    try:
        async with httpx.AsyncClient(timeout=10.0, follow_redirects=True) as client:
            response = await client.get(url, headers=headers)
        if response.status_code != 200:
            return []
        
        soup = BeautifulSoup(response.text, "html.parser")
        results = []
        for row in soup.select(".result")[:num_results]:
            title_el = row.select_one(".result__title")
            snippet_el = row.select_one(".result__snippet")
            link_el = row.select_one(".result__url")
            
            if title_el and link_el:
                title = title_el.get_text(strip=True)
                href = link_el.get("href")
                if href and href.startswith("//duckduckgo.com/l/?kh="):
                    parsed = urllib.parse.urlparse(href)
                    qs = urllib.parse.parse_qs(parsed.query)
                    href = qs.get("uddg", [href])[0]
                elif href and href.startswith("//"):
                    href = "https:" + href
                snippet = snippet_el.get_text(strip=True) if snippet_el else ""
                results.append({
                    "title": title,
                    "url": href,
                    "snippet": snippet
                })
        return results
    except Exception as e:
        print(f"[ERROR] DDG fallback search error: {e}")
        return []

async def search_web_async(query: str, num_results: int = 5) -> list[dict]:
    """
    Searches Google first, and falls back to DuckDuckGo on failure.
    """
    results = await google_search_scrape_async(query, num_results)
    if not results:
        print(f"[INFO] Google search returned no results. Falling back to DuckDuckGo...")
        results = await ddg_search_scrape_async(query, num_results)
    return results

async def fetch_url_text_async(url: str) -> str:
    """
    Fetches raw HTML of a page and parses it into clean, readable text.
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    try:
        async with httpx.AsyncClient(timeout=10.0, follow_redirects=True) as client:
            response = await client.get(url, headers=headers)
        if response.status_code != 200:
            return ""
        soup = BeautifulSoup(response.text, "html.parser")
        for element in soup(["script", "style", "nav", "footer", "header"]):
            element.decompose()
        text = soup.get_text(separator=" ")
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = "\n".join(chunk for chunk in chunks if chunk)
        # Limit size to 6000 characters to prevent context window explosion
        return text[:6000]
    except Exception as e:
        print(f"[ERROR] Fetching URL {url} error: {e}")
        return ""

def parse_json_safely(text: str) -> Any:
    """
    Parses JSON safely from LLM output, handling markdown wrappers.
    """
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()
    try:
        return json.loads(cleaned)
    except Exception as e:
        print(f"[ERROR] JSON decoding failed: {e}\nRaw output: {text}")
        return None

async def perform_angle_research(angle: str, topic: str, api_key: str | None) -> str:
    """
    Performs research on a specific angle: searches web, fetches pages, extracts cited claims.
    """
    # 1. Scrape search results
    search_results = await search_web_async(angle, num_results=3)
    if not search_results:
        return f"### Research Angle: {angle}\nNo web results found.\n"
    
    # 2. Concurrent fetches of top pages
    urls_to_fetch = [res["url"] for res in search_results[:2] if res["url"].startswith("http")]
    page_texts = await asyncio.gather(*(fetch_url_text_async(url) for url in urls_to_fetch), return_exceptions=True)
    
    fetched_data = []
    for i, url in enumerate(urls_to_fetch):
        page_text = page_texts[i]
        if isinstance(page_text, str) and page_text.strip():
            fetched_data.append({
                "title": search_results[i]["title"],
                "url": url,
                "text": page_text
            })
        else:
            # Fallback to search snippets if webpage fetch failed
            fetched_data.append({
                "title": search_results[i]["title"],
                "url": url,
                "text": f"Snippet Fallback: {search_results[i]['snippet']}"
            })
            
    # Add third source as snippet fallback
    if len(search_results) > 2:
        res = search_results[2]
        fetched_data.append({
            "title": res["title"],
            "url": res["url"],
            "text": f"Snippet Fallback: {res['snippet']}"
        })
        
    # 3. Format extraction prompt
    prompt = f"Research Topic: {topic}\nResearch Angle: {angle}\n\nGathered Sources:\n"
    for i, d in enumerate(fetched_data):
        prompt += f"--- Source [{i+1}]: {d['title']} ({d['url']}) ---\nContent:\n{d['text']}\n\n"
        
    prompt += "Extract key facts, statistics, numbers, and specific claims. Cite them with the source Title and URL."
    
    system_prompt = (
        "You are an expert research analyst. "
        "Extract key facts, statistics, numbers, and specific claims from the provided text about the research topic. "
        "For each fact/claim, map it to the source URL and source title it came from. "
        "Format your output as a list of bullet points: "
        "- [FACT/CLAIM] (Source: [Source Title](URL)) "
        "Keep it factual, objective, and precise. If no facts are found, return a message stating that."
    )
    
    extraction = await run_agent_with_key_async(system_prompt, prompt, api_key)
    return f"### Research Angle: {angle}\n\n{extraction}\n"

# =====================================================================
# MAIN DEEP RESEARCH PIPELINE
# =====================================================================

async def run_deep_research(query: str, api_key: str | None = None) -> AsyncGenerator[str, None]:
    """
    Main orchestrator for Deep Research. Streams state updates to the caller.
    """
    try:
        # Step 1: Intent & Entity Detection
        yield "🔍 Detecting query intent (stock ticker vs. general research query)..."
        
        intent_system = (
            "You are an intent and entity detector. "
            "Analyze the user's input. "
            "Determine if the input is a stock ticker symbol (e.g. AAPL, NVDA, TSLA, MSFT) or a general research query. "
            "Respond in a clean JSON format: "
            "{\n"
            "  \"is_ticker\": true/false,\n"
            "  \"ticker\": \"TICKER_NAME_IF_TRUE_ELSE_NULL\",\n"
            "  \"company_name\": \"COMPANY_NAME_IF_TRUE_ELSE_NULL\",\n"
            "  \"context\": \"Context tags or sector description (e.g., semiconductors, GPUs, AI for NVDA)\"\n"
            "}\n"
            "Do not include any markdown fences or other text."
        )
        
        intent_raw = await run_agent_with_key_async(intent_system, f"User Input: {query}", api_key)
        intent_data = parse_json_safely(intent_raw) or {"is_ticker": False, "ticker": None, "company_name": None, "context": ""}
        
        is_ticker = intent_data.get("is_ticker", False)
        company_name = intent_data.get("company_name")
        context = intent_data.get("context", "")
        ticker = intent_data.get("ticker")
        
        # Determine discovery search query
        if is_ticker and company_name:
            status_msg = f"📈 Detected stock ticker **{ticker}** ({company_name} - {context})."
            discovery_query = f"{company_name} company overview business model and performance"
        else:
            status_msg = f"🌐 Detected general query: *\"{query}\"*."
            discovery_query = query
            
        yield f"{status_msg}\n🌐 Running initial discovery search..."
        
        # Step 2: Initial discovery search
        discovery_results = await search_web_async(discovery_query, num_results=5)
        if not discovery_results:
            discovery_text = "No search snippets available."
        else:
            discovery_text = "\n".join([f"- {r['title']}: {r['snippet']}" for r in discovery_results])
            
        yield f"{status_msg}\n🧠 Generating non-overlapping research angles..."
        
        # Step 3: Generate Research Angles
        angles_system = (
            "You are a senior research coordinator. "
            "Based on the user's research topic/entity and these initial search snippets, generate 3 to 4 specific, relevant, and non-overlapping research angles (queries) to perform deep dives. "
            "Respond in a clean JSON list format: "
            "[\n"
            "  \"query 1\",\n"
            "  \"query 2\",\n"
            "  \"query 3\",\n"
            "  \"query 4\"\n"
            "]\n"
            "Ensure each query is tailored for a Google Search to get high-quality web pages. "
            "Do not include any markdown fences or other text."
        )
        
        topic_desc = f"{company_name} ({ticker})" if is_ticker else query
        angles_prompt = f"Research Topic: {topic_desc}\n\nDiscovery Snippets:\n{discovery_text}"
        angles_raw = await run_agent_with_key_async(angles_system, angles_prompt, api_key)
        angles = parse_json_safely(angles_raw)
        
        if not angles or not isinstance(angles, list):
            # Fallback queries
            if is_ticker:
                angles = [
                    f"{company_name} SWOT analysis strategic positioning",
                    f"{company_name} financial performance stock last 12 months",
                    f"{company_name} key competitors market share",
                    f"{company_name} latest quarterly results and earnings report guidance"
                ]
            else:
                angles = [
                    f"{query} current state latest developments",
                    f"{query} key challenges controversy risks",
                    f"{query} future outlook predictions market growth"
                ]
        
        angles_summary = "\n".join([f"  - **{a}**" for a in angles])
        yield f"{status_msg}\n🔍 Research Angles:\n{angles_summary}\n\n⚡ Launching parallel deep dives and scraping web pages..."
        
        # Step 4: Parallel Deep Dives
        tasks = [perform_angle_research(angle, topic_desc, api_key) for angle in angles]
        research_outputs = await asyncio.gather(*tasks)
        compiled_research = "\n\n".join(research_outputs)
        
        yield f"{status_msg}\n✍️ Synthesizing all findings and citations into a detailed report..."
        
        # Step 5: Synthesis
        synthesis_system = (
            "You are an expert research analyst. "
            "Create a comprehensive, highly detailed, and structured research report based on the provided findings from our parallel deep-dive research. "
            "You must structure the report as follows:\n"
            "1. Executive Summary: High-level overview of the findings.\n"
            "2. Detailed Findings: Create clear sections for each research angle, summarizing key points and trends.\n"
            "3. Evidence Bullets: For each section, list specific claims and numbers with inline citations to the sources (e.g. [Source Title](URL)).\n"
            "4. Risks, uncertainties, and conflicting information.\n"
            "5. 'What to watch next' checklist.\n\n"
            "Constraints & Quality Bars:\n"
            "- Prefer primary sources for financials (earnings release, filings, investor relations) and reputable outlets for news.\n"
            "- Use clean markdown tables, bold text, and bullet points.\n"
            "- Ensure every major statistic or claim is cited inline."
        )
        
        synthesis_prompt = f"Research Topic: {topic_desc}\n\nFindings from deep-dives:\n{compiled_research}"
        final_report = await run_agent_with_key_async(synthesis_system, synthesis_prompt, api_key)
        
        # Final output
        yield final_report
        
    except Exception as e:
        yield f"⚠️ An error occurred during the deep research process: {str(e)}"
