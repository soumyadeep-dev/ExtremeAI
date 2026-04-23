import sys
import asyncio
import nest_asyncio

# 1. The ultimate Windows async loop patch
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
nest_asyncio.apply()

from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from crawl4ai import AsyncWebCrawler
from ddgs import DDGS  # <-- Swapped to the NEW, actively maintained library  # <-- Swapped to the native, unblocked library

# 2. Initialize Local Model
llm = ChatOllama(model="llama3.1:8b", temperature=0.1)

async def expand_query(topic: str) -> list[str]:
    """Phase 1: Generate sub-queries strictly."""
    prompt = PromptTemplate.from_template(
        "You are an expert researcher. Break down the topic '{topic}' into 5 highly specific search queries. "
        "CRITICAL: To avoid bot-blockers, append words like 'tutorial', 'blog', or 'example' to your queries instead of looking for official docs. "
        "Respond ONLY with the queries separated by commas. Do not include introductory text."
    )
    chain = prompt | llm
    result = await chain.ainvoke({"topic": topic})
    raw_text = result.content.replace("\n", "").replace("*", "").replace("`", "")
    return [q.strip() for q in raw_text.split(",") if q.strip()]

async def extract_page_details(url: str, text: str, topic: str) -> str:
    """The Map Step: Extract dense facts from the page."""
    prompt = PromptTemplate.from_template(
        "Extract every highly detailed fact, code snippet, and feature comparison about '{topic}' from the text below.\n\nSource: {url}\n\nText: {text}"
    )
    chain = prompt | llm
    result = await chain.ainvoke({"topic": topic, "url": url, "text": text})
    return f"### Facts from {url}:\n{result.content}\n"

async def synthesize_results(topic: str, all_facts: str) -> str:
    """The Reduce Step: Final massive synthesis with Knowledge Fallback."""
    prompt = PromptTemplate.from_template(
        "You are a Principal Engineer. Write a massive, highly detailed Markdown report on '{topic}'. \n\n"
        "INSTRUCTIONS:\n"
        "1. First, attempt to use the Extracted Facts provided below.\n"
        "2. IF the Extracted Facts are empty, lack detail, or look like anti-bot warnings (e.g., 'Checking your browser'), ignore them completely. "
        "Instead, USE YOUR OWN INTERNAL EXPERT KNOWLEDGE to write the complete, detailed report.\n"
        "3. Format with deep technical sections and code blocks if applicable.\n\n"
        "Extracted Facts:\n{facts}"
    )
    chain = prompt | llm
    final_report = await chain.ainvoke({"topic": topic, "facts": all_facts})
    return final_report.content

async def run_deep_research(topic: str) -> dict:
    """The Orchestrator: The One Browser Pattern"""
    print(f"\n--- Starting DEEP Research: {topic} ---")
    
    queries = await expand_query(topic)
    print(f"Generated {len(queries)} queries. Firing up headless browser...")
    
    all_sources = []
    combined_facts = ""
    
    # ONE BROWSER INSTANCE FOR STABILITY
    async with AsyncWebCrawler() as crawler:
        for query in queries:
            print(f"\n[Searching] {query}")
            try:
                # 1. Use Native DDGS to bypass LangChain's rate limits
                with DDGS() as ddgs:
                    # Get top 2 results directly as a list of dictionaries
                    results = list(ddgs.text(query, max_results=2))
                
                # Natively grab the exact href links
                target_urls = [res.get('href') for res in results if 'href' in res]
                
                if not target_urls:
                    print("  -> 🚨 FAILED: DuckDuckGo returned no links for this query.")
                    continue
                    
                for target_url in target_urls:
                    print(f"  -> Scraping: {target_url}")
                    try:
                        scrape_result = await crawler.arun(url=target_url)
                        # Use the updated variable name
                        clean_text = scrape_result.markdown[:4000]
                        
                        # NEW: Print a preview of what the bot actually read!
                        print(f"  -> Scraped {len(clean_text)} chars. Preview: {clean_text[:100].replace(chr(10), ' ')}...")
                        
                        if clean_text:
                            print(f"  -> Extracting facts...")
                            facts = await extract_page_details(target_url, clean_text, topic)
                            combined_facts += facts + "\n"
                            all_sources.append(target_url)
                        else:
                            print(f"  -> 🚨 FAILED: Scraper returned empty text for {target_url}")
                            
                    except Exception as e:
                        print(f"  -> 🚨 Error on {target_url}: {e}")
                        
            except Exception as e:
                print(f"  -> 🚨 Search engine error for '{query}': {e}")
                    
    print("\nSynthesizing massive final report...")
    final_report = await synthesize_results(topic, combined_facts)
    
    return {
        "report": final_report,
        "queries": queries,
        "sources": list(set(all_sources)),
        "raw_data": [combined_facts]
    }