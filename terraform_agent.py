from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate

# Initialize LLM with zero temperature for high accuracy/no hallucination in code
codegen_llm = ChatOllama(model="llama3.1:8b", temperature=0.0)

async def generate_terraform(topic: str, research_data: str) -> str:
    """The Terraform Writer Agent."""
    
    prompt = PromptTemplate.from_template(
        "You are an AWS Certified Solutions Architect and Terraform Expert. "
        "Your task is to create a complete, production-ready 'main.tf' file based on the research provided below. \n\n"
        "TOPIC: {topic}\n"
        "RESEARCH DATA:\n{research}\n\n"
        "REQUIREMENTS:\n"
        "1. Include the necessary terraform and aws provider blocks.\n"
        "2. Use variables for account IDs or regions if specified.\n"
        "3. Add comments explaining each resource based on the research.\n"
        "4. Return ONLY the code inside ```hcl blocks. No conversational filler."
    )
    
    chain = prompt | codegen_llm
    result = await chain.ainvoke({"topic": topic, "research": research_data})
    return result.content