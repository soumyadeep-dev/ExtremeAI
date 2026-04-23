import asyncio
import re
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_community.tools import DuckDuckGoSearchResults
from crawl4ai import AsyncWebCrawler

# 1. Initialize Local Model via Ollama
llm = ChatOllama(model="llama3.1:8b", temperature=0.1)

# 2. Initialize Free Search
search_tool = DuckDuckGoSearchResults(backend="lite")

async def expand_query(topic: str) -> list[str]:
    """Phase 1: Break the topic into sub-queries."""
    prompt = PromptTemplate.from_template(
        "You are an expert researcher. Break down the following topic into 3 specific, highly technical search queries to find the most optimized results. Return ONLY the queries separated by commas. Topic: {topic}"
    )
    chain = prompt | llm
    result = await chain.ainvoke({"topic": topic})
    
    # Clean and split into a list
    queries = [q.strip() for q in result.content.split(",") if q.strip()]
    return queries

async def worker_search_and_scrape(query: str) -> str:
    """Phase 2: DuckDuckGo Search + Async Web Scrape."""
    try:
        # Search DDG
        raw_results = search_tool.run(query)
        
        # Extract URLs via Regex
        urls = re.findall(r'link:\s*(https?://[^\s,\]]+)', raw_results)
        if not urls:
            return f"No links found for query: {query}"
            
        target_url = urls[0] 
        print(f"Scraping: {target_url}")
        
        # Scrape headless
        async with AsyncWebCrawler() as crawler:
            scrape_result = await crawler.arun(url=target_url)
            
            # Truncated to 2000 chars to protect Llama 3.1 8k context window
            clean_text = scrape_result.markdown_links_cleaned[:2000] 
            return f"### Source: {target_url}\n{clean_text}"
            
    except Exception as e:
        return f"Error on query '{query}': {str(e)}"

async def synthesize_results(topic: str, raw_data: list[str]) -> str:
    """Phase 3: Synthesize local Markdown into one report."""
    combined_data = "\n\n".join(raw_data)
    
    prompt = PromptTemplate.from_template(
        "You are a Senior Technical Architect. Write a comprehensive Markdown report on the topic '{topic}' based ONLY on the following raw data. Do not hallucinate external facts.\n\nRaw Data:\n{data}"
    )
    chain = prompt | llm
    final_report = await chain.ainvoke({"topic": topic, "data": combined_data})
    return final_report.content

async def run_deep_research(topic: str) -> str:
    """The Orchestrator"""
    print(f"\n--- Starting Research: {topic} ---")
    
    queries = await expand_query(topic)
    print(f"Generated sub-queries: {queries}")
    
    tasks = [worker_search_and_scrape(query) for query in queries]
    scraped_results = await asyncio.gather(*tasks)
    
    print("\nSynthesizing final report...")
    final_report = await synthesize_results(topic, scraped_results)
    return final_report