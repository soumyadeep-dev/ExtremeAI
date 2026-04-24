import streamlit as st
import requests

st.set_page_config(page_title="ExtremeAI Optimized", layout="wide")

st.title("🕵️‍♂️ ExtremeAI: Parallel Deep Research")

topic = st.text_input("Enter goal:")

if st.button("Run Pipeline"):
    with st.spinner("Processing... (Parallel Workers Active)"):
        res = requests.post("http://127.0.0.1:8000/research", json={"topic": topic}).json()
        
        lang = res['language']
        ext_map = {"html-css": "html", "python": "py", "java": "java", "terraform": "tf", "react": "jsx", "go": "go"}
        ext = ext_map.get(lang, "txt")
        
        t1, t2 = st.tabs(["📄 Research", f"🛠️ {lang.upper()} Code"])
        
        with t1:
            st.markdown(res['report'])
        with t2:
            st.code(res['generated_code'], language=lang if lang != "html-css" else "html")
            st.download_button(f"Download output.{ext}", res['generated_code'], file_name=f"output.{ext}")