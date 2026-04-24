import sys
import asyncio
import nest_asyncio
from ddgs import DDGS
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig, CacheMode
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from memory import query_memory # <-- IMPORT MEMORY READER

if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
nest_asyncio.apply()

fast_llm = ChatOllama(model="llama3.2:1b", temperature=0.1) 
heavy_llm = ChatOllama(model="llama3.1:8b", temperature=0.1)
semaphore = asyncio.Semaphore(3)

async def expand_query(topic: str) -> list[str]:
    """Uses Fast Model to generate 5 robust queries."""
    prompt = PromptTemplate.from_template(
        "Break '{topic}' into 5 specific search queries. "
        "Mix query types: include 'tutorial', 'best practices', "
        "'common errors', 'example', 'vs alternatives'. "
        "Output ONLY comma-separated queries, nothing else."
    )
    chain = prompt | fast_llm
    result = await chain.ainvoke({"topic": topic})
    return [q.strip() for q in result.content.split(",") if q.strip()]

async def process_url(crawler, url, topic):
    """Worker: Parallel Scrape + Token Reduction + Extraction."""
    async with semaphore:
        try:
            config = CrawlerRunConfig(cache_mode=CacheMode.ENABLED)
            scrape_result = await crawler.arun(url=url, config=config)
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

async def synthesize_results(topic: str, combined_facts: str, existing_knowledge: str) -> str:
    """Structured technical writing using live + past data."""
    prompt = PromptTemplate.from_template("""
        You are a Principal Engineer writing a technical reference document.

        Topic: {topic}
        Existing Memory Knowledge: {existing_knowledge}
        New Web Research Facts: {facts}

        Write a highly structured technical report with these EXACT sections:
        ## Overview
        ## Key Concepts  
        ## Implementation Details
        ## Code Examples
        ## Common Pitfalls
        ## Best Practices
        ## Summary

        Rules:
        - Be deeply specific and technical.
        - Include actual code snippets where relevant.
        - If facts contradict each other, note both perspectives.
        - Combine insights from the Existing Knowledge and New Web Research seamlessly.
        - If facts are insufficient, use your internal training knowledge to fill gaps.
    """)
    chain = prompt | heavy_llm
    final_report = await chain.ainvoke({
        "topic": topic, 
        "facts": combined_facts,
        "existing_knowledge": existing_knowledge
    })
    return final_report.content

async def run_deep_research(topic: str) -> dict:
    print(f"\n--- Starting High-Speed Research: {topic} ---")
    
    # 1. READ FROM FAISS MEMORY
    existing_knowledge = query_memory(topic)
    if existing_knowledge:
        print(f"🧠 Found existing contextual memory for: {topic}")

    # 2. GENERATE UP TO 15 URLs (5 queries x 3 results)
    queries = await expand_query(topic)
    all_sources = []
    
    async with AsyncWebCrawler() as crawler:
        tasks = []
        for query in queries:
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=3)) # Boosted from 2 to 3
            
            urls = [res.get('href') for res in results if 'href' in res]
            all_sources.extend(urls)
            for url in urls:
                tasks.append(process_url(crawler, url, topic))
        
        # 3. RUN PARALLEL SCRAPERS
        fact_results = await asyncio.gather(*tasks)
        combined_facts = "\n".join([f for f in fact_results if f])

    # 4. STRUCTURED SYNTHESIS
    final_report = await synthesize_results(topic, combined_facts, existing_knowledge)
    
    return {
        "report": final_report,
        "queries": queries,
        "sources": list(set(all_sources)),
        "raw_data": [combined_facts]
    }