import sys
import asyncio
import os
from ddgs import DDGS
from openai import AsyncOpenAI
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from memory import query_memory

if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
heavy_llm = ChatOllama(model="llama3.1:8b", temperature=0.1)

_cache: dict = {}


async def expand_query(topic: str) -> list[str]:
    """GPT-4o-mini generates 3 clean search queries. Fallback to local 1B."""
    print("🤖 Expanding queries via GPT-4o-mini...")
    try:
        response = await openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Generate exactly 3 short web search queries for the given topic. "
                        "Output ONLY the 3 queries separated by commas. "
                        "Each query must be under 12 words. "
                        "No preamble, numbering, labels, or explanation. "
                        "Format: query1, query2, query3"
                    )
                },
                {"role": "user", "content": topic}
            ],
            max_tokens=100,
            temperature=0.3
        )
        raw = response.choices[0].message.content or ""
        queries = [q.strip().strip('"') for q in raw.split(",") if 5 < len(q.strip()) < 150][:3]
        print(f"🔍 Queries: {queries}")
        return queries if queries else [topic[:100]]
    except Exception as e:
        print(f"⚠️ OpenAI query expansion failed: {e} — falling back to local")
        return await _expand_query_local(topic)


async def _expand_query_local(topic: str) -> list[str]:
    fast_llm = ChatOllama(model="llama3.2:1b", temperature=0.1)
    prompt = PromptTemplate.from_template(
        "Generate 3 search queries for: {topic}\n"
        "Output ONLY comma-separated queries. No labels.\n"
        "Format: query1, query2, query3"
    )
    result = await (prompt | fast_llm).ainvoke({"topic": topic})
    queries = [q.strip() for q in result.content.split(",") if 5 < len(q.strip()) < 150]
    return queries[:3] if queries else [topic[:100]]


def _ddg_search(query: str, max_results: int = 3) -> tuple[list[str], list[str]]:
    snippets, sources = [], []
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
        for r in results:
            if r.get("body"):
                snippets.append(f"### {r.get('title', 'Source')}\n{r['body']}")
            if r.get("href"):
                sources.append(r["href"])
    except Exception as e:
        print(f"⚠️ DDG failed for '{query}': {e}")
    return snippets, sources


async def fetch_all_snippets(queries: list[str]) -> tuple[list[str], list[str]]:
    """All DDG searches fire in parallel."""
    loop = asyncio.get_event_loop()
    tasks = [loop.run_in_executor(None, _ddg_search, q) for q in queries]
    results = await asyncio.gather(*tasks)
    all_snippets, all_sources = [], []
    for snippets, sources in results:
        all_snippets.extend(snippets)
        all_sources.extend(sources)
    return all_snippets, all_sources


async def synthesize_results(topic: str, combined_facts: str, existing_knowledge: str) -> str:
    """Local 8B synthesis — research data stays private, never sent to OpenAI."""
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
        - If facts contradict, note both perspectives.
        - Combine Existing Knowledge and New Research seamlessly.
        - Fill gaps using your internal training knowledge.
    """)
    result = await (prompt | heavy_llm).ainvoke({
        "topic": topic,
        "facts": combined_facts,
        "existing_knowledge": existing_knowledge
    })
    return result.content


async def run_deep_research(topic: str) -> dict:
    print(f"\n--- Starting Research: {topic} ---")

    cache_key = topic.strip().lower()
    if cache_key in _cache:
        print("⚡ Cache hit")
        return _cache[cache_key]

    # Step 1: Memory + query expansion in parallel
    loop = asyncio.get_event_loop()
    existing_knowledge, queries = await asyncio.gather(
        loop.run_in_executor(None, query_memory, topic),
        expand_query(topic)
    )

    if existing_knowledge:
        print("🧠 Found existing memory for topic")

    # Step 2: Parallel DDG fetches
    snippets, all_sources = await fetch_all_snippets(queries)
    print(f"📄 {len(snippets)} snippets from {len(all_sources)} sources")

    # Step 3: Synthesize (local 8B, stays private)
    combined_facts = "\n\n".join(snippets)[:6000]
    final_report = await synthesize_results(topic, combined_facts, existing_knowledge)

    # Step 4: Reflection — model critiques its own report
    # Import here to avoid circular imports
    from coder_agent import reflect_on_report
    reflection = await reflect_on_report(topic, final_report)
    final_report = final_report + "\n\n" + reflection

    result = {
        "report": final_report,
        "queries": queries,
        "sources": list(set(all_sources)),
        "raw_data": [combined_facts]
    }

    _cache[cache_key] = result
    return result