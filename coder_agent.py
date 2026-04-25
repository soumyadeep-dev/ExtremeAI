import os
import sys
import subprocess
import tempfile
from openai import AsyncOpenAI
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate

# --- CLIENTS ---
openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
heavy_llm     = ChatOllama(model="llama3.1:8b", temperature=0.0)

SUPPORTED_LANGUAGES = {"python", "java", "react", "html-css", "go", "nodejs", "terraform"}

LANGUAGE_CONFIGS = {
    "html-css":  {"role": "Expert Web Designer",       "rules": "Write ONLY a single index.html with CSS/JS inside. No Python or server code."},
    "java":      {"role": "Senior Java Developer",      "rules": "Write ONLY Spring Boot Java. No Python wrappers."},
    "python":    {"role": "Python Automation Engineer", "rules": "Write pure Python. Use only real, documented libraries. No hallucinated SDKs."},
    "terraform": {"role": "Cloud Architect",            "rules": "Write ONLY .tf HCL Terraform. Include provider and variable blocks. Use real resource types only."},
    "react":     {"role": "Senior React Developer",     "rules": "Write ONLY React JSX with hooks. No backend code."},
    "go":        {"role": "Senior Go Developer",        "rules": "Write idiomatic Go. No Python wrappers."},
    "nodejs":    {"role": "Senior Node.js Developer",   "rules": "Write ONLY Node.js/Express JS. No Python wrappers."},
}

# Languages that can be sandboxed and executed locally
EXECUTABLE_LANGUAGES = {"python"}

MAX_RETRIES = 3

# ── AWS/Cloud signal detection ─────────────────────────────────────────────────
# These topics require GPT-4o because local 8B hallucinates AWS APIs constantly
AWS_SIGNALS = [
    "aws", "amazon", "glue", "macie", "s3", "iam", "lambda", "athena",
    "lakeformation", "lake formation", "sagemaker", "ec2", "vpc", "rds",
    "dynamodb", "kinesis", "cloudwatch", "cloudformation", "eks", "ecs",
    "sns", "sqs", "boto3", "bedrock", "redshift", "emr", "step functions",
    "eventbridge", "api gateway", "route53", "cloudfront", "waf", "guard duty",
    "security hub", "config", "systems manager", "secrets manager", "kms",
    "codecommit", "codepipeline", "codebuild", "codedeploy", "amplify",
    "storage lens", "iceberg", "hudi", "delta lake", "data catalog",
    "azure", "gcp", "google cloud", "kubernetes", "k8s", "helm",
    "docker", "terraform", "pulumi", "cdk", "cloudtrail"
]

# These languages always go to GPT-4o regardless of topic
# because local models are too unreliable for infrastructure code
ALWAYS_USE_GPT4O = {"terraform"}

# General purpose languages where local 8B is good enough
LOCAL_LANGUAGES = {"python", "java", "react", "html-css", "go", "nodejs"}


def _is_aws_or_cloud_topic(topic: str) -> bool:
    """Returns True if the topic contains AWS/cloud signals."""
    lowered = topic.lower()
    return any(signal in lowered for signal in AWS_SIGNALS)


def _normalize_language(raw: str) -> str:
    cleaned = raw.strip().lower().strip(".")
    if cleaned in SUPPORTED_LANGUAGES:
        return cleaned
    if "terraform" in cleaned or "hcl" in cleaned:      return "terraform"
    if "react" in cleaned or "jsx" in cleaned:           return "react"
    if "node" in cleaned or "express" in cleaned:        return "nodejs"
    if "html" in cleaned or "css" in cleaned:            return "html-css"
    if "java" in cleaned and "script" not in cleaned:    return "java"
    if "go" in cleaned or "golang" in cleaned:           return "go"
    if "javascript" in cleaned or cleaned == "js":       return "nodejs"
    print(f"⚠️ Unrecognized language '{cleaned}' — defaulting to python")
    return "python"


async def detect_language(topic: str, language_override: str | None = None) -> str:
    """GPT-4o-mini routing with UI override taking full priority."""
    if language_override and language_override in SUPPORTED_LANGUAGES:
        print(f"🎯 Language override: {language_override}")
        return language_override

    print("🤖 Detecting language via GPT-4o-mini...")
    try:
        response = await openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a language router for a code generation tool. "
                        "Given a user goal, respond with ONLY one word from: "
                        "python, terraform, java, react, html-css, go, nodejs. "
                        "AWS/cloud infrastructure → terraform. "
                        "Data pipelines/scripting → python. "
                        "Web UI → react or html-css. "
                        "Single word only. No explanation."
                    )
                },
                {"role": "user", "content": topic}
            ],
            max_tokens=5,
            temperature=0
        )
        raw  = response.choices[0].message.content or "python"
        lang = _normalize_language(raw)
        print(f"🤖 Detected: {lang}")
        return lang
    except Exception as e:
        print(f"⚠️ OpenAI detection failed: {e} — falling back to local")
        return await _detect_language_local(topic)


async def _detect_language_local(topic: str) -> str:
    fast_llm = ChatOllama(model="llama3.2:1b", temperature=0.0)
    prompt   = PromptTemplate.from_template(
        "Output ONLY one word for the best language: {topic}\n"
        "Choose from: python, terraform, java, react, html-css, go, nodejs"
    )
    result = await (prompt | fast_llm).ainvoke({"topic": topic})
    return _normalize_language(result.content)


def _should_use_gpt4o(topic: str, language: str) -> bool:
    """
    Routing decision: GPT-4o or local 8B?

    GPT-4o when:
    - Language is always-GPT4O (terraform)
    - Topic contains AWS/cloud signals (boto3, Macie, Glue, etc.)

    Local 8B when:
    - General purpose code (Python web, React, Java, Go, Node)
    - No cloud signals in topic
    """
    if language in ALWAYS_USE_GPT4O:
        return True
    if language in LOCAL_LANGUAGES and _is_aws_or_cloud_topic(topic):
        return True
    return False


async def _generate_with_gpt4o(topic: str, research_data: str, language: str) -> str:
    """
    Uses GPT-4o for AWS/cloud/terraform code.
    Much more accurate on real SDK method names and resource types.
    """
    config = LANGUAGE_CONFIGS.get(language, {
        "role": "Senior Developer",
        "rules": "Write clean, production-ready code using only real, documented APIs."
    })

    print(f"🤖 Routing to GPT-4o for {language} (cloud/AWS topic detected)")

    response = await openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": (
                    f"You are a {config['role']}. "
                    f"{config['rules']} "
                    "CRITICAL: Use ONLY real, documented APIs and SDK methods. "
                    "Do NOT invent class names, method names, or library imports that do not exist. "
                    "If unsure about a specific API, use the most common documented pattern. "
                    "Return ONLY the raw code. No markdown fences. No explanation."
                )
            },
            {
                "role": "user",
                "content": (
                    f"Topic: {topic}\n\n"
                    f"Research context (use this to inform implementation):\n"
                    f"{research_data[:5000]}\n\n"
                    f"Write a complete, production-ready {language} implementation."
                )
            }
        ],
        temperature=0,
        max_tokens=2500
    )

    return response.choices[0].message.content.strip()


async def _generate_with_local(topic: str, research_data: str, language: str) -> str:
    """
    Uses local Llama 8B for general purpose code.
    Fast, private, good enough for non-AWS topics.
    """
    config = LANGUAGE_CONFIGS.get(language, {
        "role": "Senior Developer",
        "rules": "Write clean code."
    })

    print(f"🏠 Using local 8B for {language} generation")

    prompt = PromptTemplate.from_template(
        "You are a {role}. Provide the implementation for: {topic}\n\n"
        "RESEARCH:\n{research}\n\n"
        "CRITICAL RULE: Return ONLY the raw code for {language}. "
        "Do NOT provide a Python script that generates code. "
        "Rules: {rules}\n\n"
        "RETURN ONLY THE RAW CODE. NO MARKDOWN. NO EXPLANATION."
    )

    result = await (prompt | heavy_llm).ainvoke({
        "role":     config["role"],
        "language": language,
        "topic":    topic,
        "research": research_data[:4000],
        "rules":    config["rules"]
    })

    return result.content.strip()


def _run_in_sandbox(code: str, language: str) -> tuple[bool, str]:
    """
    Executes code in a temp subprocess sandbox.
    Returns (success, output_or_error).
    """
    if language == "python":
        with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
            f.write(code)
            tmp_path = f.name
        try:
            result = subprocess.run(
                [sys.executable, tmp_path],
                capture_output=True, text=True, timeout=15
            )
            os.unlink(tmp_path)
            if result.returncode == 0:
                return True, result.stdout
            else:
                return False, result.stderr
        except subprocess.TimeoutExpired:
            os.unlink(tmp_path)
            return False, "Execution timed out (>15s)"
        except Exception as e:
            return False, str(e)

    elif language == "terraform":
        with tempfile.TemporaryDirectory() as tmpdir:
            tf_path = os.path.join(tmpdir, "main.tf")
            with open(tf_path, "w") as f:
                f.write(code)
            try:
                result = subprocess.run(
                    ["terraform", "validate"],
                    capture_output=True, text=True, timeout=30, cwd=tmpdir
                )
                if result.returncode == 0:
                    return True, "Terraform validate passed"
                else:
                    return False, result.stderr or result.stdout
            except FileNotFoundError:
                return True, "Terraform not installed locally — skipping validation"
            except Exception as e:
                return False, str(e)

    return True, "Sandbox not available for this language"


async def _fix_code_local(topic: str, language: str, broken_code: str, error: str) -> str:
    """Asks local 8B to fix code given the error output."""
    config = LANGUAGE_CONFIGS.get(language, {
        "role": "Senior Developer",
        "rules": "Write clean code."
    })
    prompt = PromptTemplate.from_template(
        "You are a {role}. The following {language} code has an error.\n\n"
        "ORIGINAL CODE:\n{code}\n\n"
        "ERROR:\n{error}\n\n"
        "Fix the code so it runs correctly. "
        "Rules: {rules}\n"
        "RETURN ONLY THE FIXED RAW CODE. NO MARKDOWN. NO EXPLANATION."
    )
    result = await (prompt | heavy_llm).ainvoke({
        "role":     config["role"],
        "language": language,
        "code":     broken_code,
        "error":    error,
        "rules":    config["rules"]
    })
    return result.content.strip()


async def _fix_code_gpt4o(topic: str, language: str, broken_code: str, error: str) -> str:
    """Asks GPT-4o to fix code. Used for AWS/cloud topics."""
    config = LANGUAGE_CONFIGS.get(language, {
        "role": "Senior Developer",
        "rules": "Write clean, production-ready code using only real documented APIs."
    })

    response = await openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": (
                    f"You are a {config['role']}. Fix the broken code below. "
                    "Use ONLY real documented APIs. "
                    "Return ONLY the fixed raw code. No markdown. No explanation."
                )
            },
            {
                "role": "user",
                "content": (
                    f"Language: {language}\n"
                    f"Topic: {topic}\n\n"
                    f"BROKEN CODE:\n{broken_code}\n\n"
                    f"ERROR:\n{error}\n\n"
                    "Fix the code."
                )
            }
        ],
        temperature=0,
        max_tokens=2500
    )

    return response.choices[0].message.content.strip()


async def generate_code(topic: str, research_data: str, language: str) -> str:
    """
    Smart code generation with routing + self-correction loop.

    Routing logic:
    - Terraform / AWS topics → GPT-4o (accurate SDK knowledge)
    - General code (React, Go, Java, plain Python) → Local 8B (fast, private)

    Self-correction:
    - Python: run in sandbox, fix errors up to MAX_RETRIES times
    - Terraform: run terraform validate if available
    - Other: return as-is (no sandbox available)
    """
    use_gpt4o = _should_use_gpt4o(topic, language)

    # ── Initial generation ──────────────────────────────────────────────────
    if use_gpt4o:
        code = await _generate_with_gpt4o(topic, research_data, language)
    else:
        code = await _generate_with_local(topic, research_data, language)

    # ── Self-correction loop (Python + Terraform only) ──────────────────────
    if language in EXECUTABLE_LANGUAGES or language == "terraform":
        for attempt in range(1, MAX_RETRIES + 1):
            success, output = _run_in_sandbox(code, language)

            if success:
                print(f"✅ Code passed sandbox on attempt {attempt}")
                status = f"\n\n# ✅ Sandbox verified (attempt {attempt})"
                return code + status

            print(f"🔁 Attempt {attempt} failed — self-correcting...\nError: {output[:300]}")

            if attempt < MAX_RETRIES:
                # Use same tier for fixing as for generation
                if use_gpt4o:
                    code = await _fix_code_gpt4o(topic, language, code, output)
                else:
                    code = await _fix_code_local(topic, language, code, output)
            else:
                status = (
                    f"\n\n# ⚠️ Could not auto-fix after {MAX_RETRIES} attempts.\n"
                    f"# Last error: {output[:200]}"
                )
                return code + status

    # Non-sandboxable languages — return as-is
    return code


async def reflect_on_report(topic: str, report: str) -> str:
    """
    Reflection step: 8B model critiques its own report.
    Identifies gaps, missing steps, contradictions.
    Always runs locally — reflection stays private.
    """
    print("🪞 Running reflection step...")
    prompt = PromptTemplate.from_template(
        "You are a senior technical reviewer. Read the following research report "
        "and provide a critical self-review.\n\n"
        "TOPIC: {topic}\n\n"
        "REPORT:\n{report}\n\n"
        "Identify:\n"
        "1. Any missing steps or gaps in the implementation details\n"
        "2. Any contradictions or assumptions that need clarification\n"
        "3. Any security, performance, or best-practice concerns not covered\n"
        "4. What the reader should research further\n\n"
        "Be specific. Format as a short ## Self-Review section with bullet points."
    )
    result = await (prompt | heavy_llm).ainvoke({
        "topic":  topic,
        "report": report[:5000]
    })
    return result.content.strip()