from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate

fast_llm = ChatOllama(model="llama3.2:1b", temperature=0.0)
heavy_llm = ChatOllama(model="llama3.1:8b", temperature=0.0)

async def detect_language(topic: str) -> str:
    """Instant Routing using 1B Model."""
    prompt = PromptTemplate.from_template(
        "Based on the user goal: '{topic}', output ONLY the language name: "
        "(java, python, react, html-css, go, nodejs, terraform)."
    )
    chain = prompt | fast_llm
    result = await chain.ainvoke({"topic": topic})
    return result.content.strip().lower()

async def generate_code(topic: str, research_data: str, language: str) -> str:
    """Zero-Wrapper Code Generation using 8B Model."""
    
    # Logic to fix the "HTML in Python" and "Terraform in Python" bug
    configs = {
        "html-css": {
            "role": "Expert Web Designer",
            "rules": "Write ONLY a single index.html file with CSS/JS inside. Do NOT write Python, Flask, or server code."
        },
        "java": {
            "role": "Senior Java Developer",
            "rules": "Write ONLY Spring Boot Java code. No Python wrappers."
        },
        "python": {
            "role": "Python Automation Engineer",
            "rules": "Write pure Python code for automation or AWS Lambda."
        },
        "terraform": {
            "role": "Cloud Architect",
            "rules": "Write ONLY .tf HCL code. No Python wrappers."
        }
    }
    
    config = configs.get(language, {"role": "Senior Developer", "rules": "Write clean code."})

    prompt = PromptTemplate.from_template(
        "You are a {role}. Provide the implementation for: {topic}\n\n"
        "RESEARCH:\n{research}\n\n"
        "CRITICAL RULE: Return ONLY the raw code for {language}. "
        "Do NOT provide a Python script that generates this code. "
        "Do NOT provide a backend wrapper unless requested.\n\n"
        "Rules: {rules}\n\n"
        "RETURN ONLY THE RAW CODE. NO MARKDOWN. NO EXPLANATION."
    )
    
    chain = prompt | heavy_llm
    result = await chain.ainvoke({
        "role": config["role"],
        "language": language,
        "topic": topic,
        "research": research_data,
        "rules": config["rules"]
    })
    
    return result.content.strip()