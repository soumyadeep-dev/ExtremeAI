import streamlit as st
import requests
import json

st.set_page_config(page_title="ExtremeAI | Polyglot Agent", page_icon="🕵️‍♂️", layout="wide")
st.title("🕵️‍♂️ ExtremeAI: Deep Research + Code Generation")

# --- INPUT ROW ---
col1, col2 = st.columns([4, 1])
with col1:
    topic = st.text_input(
        "Enter your project goal or research topic:",
        placeholder="e.g., Build a Python web scraper using BeautifulSoup"
    )
with col2:
    lang_override = st.selectbox(
        "Language",
        options=["Auto-Detect", "python", "terraform", "java", "react", "html-css", "go", "nodejs"],
        index=0,
        help="Set manually for AWS/cloud topics — Auto-Detect can misfire on infrastructure goals."
    )

use_streaming = st.toggle("⚡ Enable Streaming (see output live)", value=True)

# Warn if infra topic + auto-detect
INFRA_KEYWORDS = [
    "lakeformation", "lake formation", "glue", "iam", "s3", "ec2",
    "vpc", "cloudformation", "aws", "azure", "gcp", "eks", "ecs",
    "terraform", "rds", "sqs", "sns", "lambda", "macie", "athena"
]
if topic and lang_override == "Auto-Detect":
    if any(kw in topic.lower() for kw in INFRA_KEYWORDS):
        st.warning("⚠️ Infrastructure topic detected — set **Language → terraform** or **python** to avoid misdetection.")

EXT_MAP = {
    "html-css": "html", "python": "py",
    "java": "java", "terraform": "tf",
    "react": "jsx", "go": "go", "nodejs": "js"
}


def _build_payload(topic: str, lang_override: str) -> dict:
    return {
        "topic": topic,
        "language_override": None if lang_override == "Auto-Detect" else lang_override
    }


def run_standard(topic: str, lang_override: str):
    payload = _build_payload(topic, lang_override)
    with st.spinner("🔍 Researching and generating code..."):
        try:
            res = requests.post(
                "http://127.0.0.1:8000/research",
                json=payload,
                timeout=180
            ).json()
        except requests.exceptions.Timeout:
            st.error("⏱️ Request timed out. Try enabling streaming mode.")
            return
        except Exception as e:
            st.error(f"❌ Backend Error: {e}")
            return

    lang = res.get("language", "python")
    ext = EXT_MAP.get(lang, "txt")
    _render_results(res, lang, ext)


def run_streaming(topic: str, lang_override: str):
    payload = _build_payload(topic, lang_override)

    status_box = st.empty()
    code_placeholder = st.empty()
    status_box.info("⏳ Starting pipeline...")

    full_code = ""
    report = ""
    language = lang_override if lang_override != "Auto-Detect" else "python"
    queries, sources = [], []

    try:
        with requests.post(
            "http://127.0.0.1:8000/research/stream",
            json=payload,
            stream=True,
            timeout=180
        ) as resp:
            for line in resp.iter_lines():
                if not line:
                    continue
                decoded = line.decode("utf-8")
                if not decoded.startswith("data: "):
                    continue
                try:
                    event = json.loads(decoded[6:])
                except json.JSONDecodeError:
                    continue

                etype = event.get("type")

                if etype == "meta":
                    language = event.get("language", language)
                    queries = event.get("queries", [])
                    sources = event.get("sources", [])
                    status_box.success(f"✅ Language: **{language.upper()}** | Queries: {len(queries)} | Sources: {len(sources)}")

                elif etype == "code_chunk":
                    full_code += event.get("token", "")
                    code_placeholder.code(full_code, language=language if language != "html-css" else "html")

                elif etype == "report":
                    report = event.get("report", "")
                    status_box.success("✅ Pipeline complete!")

                elif etype == "done":
                    break

                elif etype == "error":
                    st.error(f"❌ Stream error: {event.get('detail')}")
                    return

    except requests.exceptions.Timeout:
        st.error("⏱️ Stream timed out.")
        return
    except Exception as e:
        st.error(f"❌ Streaming Error: {e}")
        return

    ext = EXT_MAP.get(language, "txt")
    _render_results({
        "report": report,
        "language": language,
        "generated_code": full_code,
        "queries": queries,
        "sources": sources,
        "raw_data": []
    }, language, ext)


def _render_results(res: dict, lang: str, ext: str):
    with st.expander("🔍 Search Queries Used"):
        for q in res.get("queries", []):
            st.markdown(f"- `{q}`")

    st.markdown("---")
    t1, t2, t3, t4 = st.tabs(["📄 Research Report", "🛠️ Generated Code", "🌐 Sources", "🗃️ Raw Data"])

    with t1:
        st.markdown(res.get("report", "No report generated."))

    with t2:
        code = res.get("generated_code", "")
        st.code(code, language=lang if lang != "html-css" else "html")
        if code:
            st.download_button(f"⬇️ Download output.{ext}", code, file_name=f"output.{ext}")

    with t3:
        sources = res.get("sources", [])
        if sources:
            for src in sources:
                st.markdown(f"- [{src}]({src})")
        else:
            st.info("No sources returned.")

    with t4:
        for i, chunk in enumerate(res.get("raw_data", [])):
            st.text_area("Raw Snippet Data", chunk, height=300, key=f"raw_{i}")


# --- RUN ---
if st.button("🚀 Run Pipeline", type="primary"):
    if not topic.strip():
        st.warning("Please enter a topic.")
    elif use_streaming:
        run_streaming(topic, lang_override)
    else:
        run_standard(topic, lang_override)