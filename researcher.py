import sys
import asyncio
import nest_asyncio
from ddgs import DDGS
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig, CacheMode
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate

# Windows async loop patch
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
nest_asyncio.apply()

# --- MODEL TIERING ---
# Fast model for logic, Heavy model for intelligence
fast_llm = ChatOllama(model="llama3.2:1b", temperature=0.1) 
heavy_llm = ChatOllama(model="llama3.1:8b", temperature=0.1)

# --- PARALLELISM ---
# Limit concurrent browser tasks to 3 to protect your local VRAM
semaphore = asyncio.Semaphore(3)

async def expand_query(topic: str) -> list[str]:
    """Uses Fast Model (1B) to generate queries instantly."""
    prompt = PromptTemplate.from_template(
        "Break '{topic}' into 3 specific search queries. "
        "Append 'tutorial' or 'blog' to queries. Output ONLY comma-separated queries."
    )
    chain = prompt | fast_llm
    result = await chain.ainvoke({"topic": topic})
    return [q.strip() for q in result.content.split(",") if q.strip()]

async def process_url(crawler, url, topic):
    """Worker: Parallel Scrape + Token Reduction + Extraction."""
    async with semaphore:
        try:
            # Token Reduction: Use Cache and strip navigation
            config = CrawlerRunConfig(cache_mode=CacheMode.ENABLED)
            scrape_result = await crawler.arun(url=url, config=config)
            
            # Reduce tokens: keep only the first 3500 chars of cleaned markdown
            clean_text = scrape_result.markdown[:3500] 
            
            if clean_text:
                prompt = PromptTemplate.from_template(
                    "Extract technical facts and code snippets about '{topic}' from: {text}"
                )
                chain = prompt | heavy_llm 
                result = await chain.ainvoke({"topic": topic, "text": clean_text})
                return f"### Facts from {url}:\n{result.content}\n"
            return ""
        except Exception:
            return ""

async def run_deep_research(topic: str) -> dict:
    print(f"\n--- Starting High-Speed Research: {topic} ---")
    queries = await expand_query(topic)
    all_sources = []
    
    async with AsyncWebCrawler() as crawler:
        tasks = []
        for query in queries:
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=2))
            
            urls = [res.get('href') for res in results if 'href' in res]
            all_sources.extend(urls)
            for url in urls:
                # Parallel Execution
                tasks.append(process_url(crawler, url, topic))
        
        # Run all scrapes and extractions at once
        fact_results = await asyncio.gather(*tasks)
        combined_facts = "\n".join([f for f in fact_results if f])

    # Final Synthesis with heavy model
    prompt = PromptTemplate.from_template(
        "Write a massive, detailed technical report on '{topic}' using only these facts:\n{facts}"
    )
    chain = prompt | heavy_llm
    final_report = await chain.ainvoke({"topic": topic, "facts": combined_facts})
    
    return {
        "report": final_report.content,
        "queries": queries,
        "sources": list(set(all_sources)),
        "raw_data": [combined_facts]
    }