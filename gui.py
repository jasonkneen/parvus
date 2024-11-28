import streamlit as st
import json
import pandas as pd
import numpy as np
from parvus import ParvusCompressor

# Initialize Compression System
if "system" not in st.session_state:
    st.session_state.system = ParvusCompressor()
system = st.session_state.system

is_compressed = False  # Flag to track compression state
# Save compressed state
def save_compressed_state():
    try:
        system.save_state("models/compressed_data.pkl", "models/faiss_index.bin")
        st.success("Saved compression state.")
    except Exception as e:
        st.error(f"Error saving compression state: {e}")

# Load compressed state
def load_compressed_state():
    global is_compressed
    if st.session_state.get("compressed_loaded", False):
        return  # Avoid reloading in the same session
    try:
        system.load_state("models/compressed_data.pkl", "models/faiss_index.bin")
        is_compressed = True
        st.session_state["compressed_loaded"] = True
        st.success("Loaded pre-compressed data successfully.")
    except FileNotFoundError:
        st.info("No saved compression state found. Please upload a dataset.")
    except Exception as e:
        st.error(f"Error loading pre-compressed state: {e}")

# Load state on app startup
load_compressed_state()

# Title and Description
st.title("Quantum Compression Dashboard")
st.write("Upload a dataset, compress it, and perform similarity queries!")

# File Upload
uploaded_file = st.file_uploader("Upload a dataset (CSV or JSON)", type=["csv", "json"])

if uploaded_file:
    try:
        if uploaded_file.name.endswith(".csv"):
            data = pd.read_csv(uploaded_file).to_numpy()
            if data.size == 0:
                st.error("Uploaded CSV file is empty.")
                st.stop()
        elif uploaded_file.name.endswith(".json"):
            data = json.loads(uploaded_file.getvalue().decode("utf-8"))
            if not data:
                st.error("Uploaded JSON file is empty.")
                st.stop()
            
            # Store the original messages in the system
            system.original_messages = [{"role": item["role"], "message": item["message"]} for item in data]
            
            # Generate embeddings
            embeddings = np.array([system.text_embedding_model.encode(item["message"]) for item in data])
            
            print(f"Loaded {len(system.original_messages)} messages")
            print("Embeddings shape:", embeddings.shape)

        # Compress dataset
        if not is_compressed:
            system.compress(embeddings)
            is_compressed = True
            save_compressed_state()

            # Display compression details
            compression_details = {
                "Original Dimension": system.decompressor.components_.shape[1],
                "Compressed Dimension": system.compressed_data.shape[1],
                "Number of Items": len(system.original_messages),
                "Compression Ratio": float(system.decompressor.components_.shape[1] / system.compressed_data.shape[1])
            }
            st.json(compression_details)
    except Exception as e:
        st.error(f"Error processing uploaded file: {e}")

# Query Section
query_input = st.text_area("Enter a query (text or numeric vector):")
top_k = st.slider("Select the number of results (top_k):", min_value=1, max_value=10, value=5)

if query_input:
    try:
        if "," in query_input:  # Numeric query 
            query_vector = np.array([float(x.strip()) for x in query_input.split(",")])
            if query_vector.size == 0:
                st.error("Numeric query vector is empty. Please provide a valid vector.")
            else:
                results, distances = system.query(query_vector, top_k=top_k, is_text_query=False)
        else:  # Text query
            if len(query_input.strip()) == 0:
                st.error("Text query is empty. Please provide a valid query.")
            else:
                results, distances = system.query(query_input, top_k=top_k, is_text_query=True)

        if results:
            # Display results
            st.write("Query Results:")
            results_df = pd.DataFrame({
                "Role": [r["role"] for r in results],
                "Message": [r["message"] for r in results],
                "Distance": distances
            })
            st.dataframe(results_df)
        else:
            st.warning("No matching results found for your query.")
    except Exception as e:
        st.error(f"Error processing query: {str(e)}")