import streamlit as st
import requests

# 1. Page Configuration
st.set_page_config(
    page_title="Deep Research Agent",
    page_icon="🕵️‍♂️",
    layout="wide"
)

# Backend API URL (Make sure FastAPI is running on this port)
API_URL = "http://127.0.0.1:8000/research"

# 2. UI Layout
st.title("🕵️‍♂️ Local Deep Research AI")
st.markdown("Enter a technical topic below. The multi-agent system will generate sub-queries, search the web, scrape the data, and synthesize a final report using your local Llama 3.1 model.")

# Create a clean input box
topic = st.text_input("What do you want to research?", placeholder="e.g., Compare Apache Iceberg vs Delta Lake performance 2026")

# 3. Action Logic
if st.button("Run Deep Research", type="primary"):
    if not topic:
        st.warning("Please enter a topic to research.")
    else:
        # Use a spinner to show the user that the backend is working
        with st.spinner(f"Agents are actively researching: '{topic}'... This may take a minute on local hardware."):
            try:
                # Send the POST request to your FastAPI backend
                response = requests.post(API_URL, json={"topic": topic})
                
                if response.status_code == 200:
                    data = response.json()
                    
                    # 4. Display Results
                    st.success(f"Research Complete! Status: {data['status']}")
                    
                    st.divider()
                    
                    # Streamlit natively renders the Markdown string returned by Llama 3.1
                    st.markdown(data['report'])
                    
                    # Optional: Expandable section to see the raw JSON payload
                    with st.expander("View Raw API Response"):
                        st.json(data)
                        
                else:
                    st.error(f"Backend Error: {response.status_code}")
                    st.write(response.text)
                    
            except requests.exceptions.ConnectionError:
                st.error("🚨 Could not connect to the backend. Is your FastAPI server running on http://127.0.0.1:8000 ?")
            except Exception as e:
                st.error(f"An unexpected error occurred: {str(e)}")