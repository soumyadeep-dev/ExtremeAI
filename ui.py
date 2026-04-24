import streamlit as st
import requests

st.set_page_config(page_title="ExtremeAI | Polyglot Agent", page_icon="🕵️‍♂️", layout="wide")
st.title("🕵️‍♂️ ExtremeAI: Deep Research + Code Generation")

topic = st.text_input("Enter your project goal or research topic:", placeholder="e.g., Build a Python web scraper using BeautifulSoup")

if st.button("🚀 Run Pipeline", type="primary"):
    if not topic:
        st.warning("Please enter a topic.")
    else:
        with st.spinner("Agents are researching, scraping, and coding. This takes 30-60 seconds..."):
            try:
                res = requests.post(
                    "http://127.0.0.1:8000/research", 
                    json={"topic": topic}
                ).json()
                
                lang = res.get('language', 'txt')
                ext_map = {
                    "html-css": "html", "python": "py", 
                    "java": "java", "terraform": "tf", 
                    "react": "jsx", "go": "go", "nodejs": "js"
                }
                ext = ext_map.get(lang, "txt")

                # Header summary
                st.success(f"Pipeline Complete! Language Detected: **{lang.upper()}**")
                
                # Show queries used
                with st.expander("🔍 AI Search Queries Generated"):
                    for q in res.get('queries', []):
                        st.markdown(f"- `{q}`")
                
                st.markdown("---")

                # Expanded Tabs
                t1, t2, t3, t4 = st.tabs([
                    "📄 Research Report", 
                    f"🛠️ Engineered Code",
                    "🌐 Sources Scraped",
                    "🗃️ Raw Data Dump"
                ])
                
                with t1:
                    st.markdown(res['report'])
                
                with t2:
                    st.code(
                        res['generated_code'], 
                        language=lang if lang != "html-css" else "html"
                    )
                    st.download_button(
                        f"⬇️ Download output.{ext}", 
                        res['generated_code'], 
                        file_name=f"output.{ext}"
                    )
                
                with t3:
                    st.markdown("### Web Sources Extracted")
                    sources = res.get('sources', [])
                    if sources:
                        for src in sources:
                            st.markdown(f"- [{src}]({src})")
                    else:
                        st.info("No external sources were successfully scraped.")
                
                with t4:
                    st.markdown("### Raw Markdown from Crawler")
                    for i, chunk in enumerate(res.get('raw_data', [])):
                        st.text_area(f"Worker Extracted Data", chunk, height=300, key=f"raw_{i}")
                        
            except Exception as e:
                st.error(f"Backend Error: {e}")