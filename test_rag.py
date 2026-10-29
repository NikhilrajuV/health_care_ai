
import streamlit as st
from src.rag import build_faiss_index, retrieve_medical_context

st.title("HealthCare AI — Medical RAG Test")

st.write(
    "Build a searchable index from the medical PDFs "
    "inside data/medical_docs."
)

if st.button("Build Medical Knowledge Base"):
    try:
        with st.spinner("Reading PDFs and generating embeddings..."):
            pdf_count, chunk_count = build_faiss_index()

        st.success(
            f"Indexed {pdf_count} PDF(s) into {chunk_count} text chunks."
        )

    except Exception as exc:
        st.error(f"Index building failed: {type(exc).__name__}: {exc}")

question = st.text_input(
    "Ask a question about your uploaded medical documents"
)

if st.button("Search Documents"):
    if not question.strip():
        st.warning("Enter a question first.")

    else:
        try:
            with st.spinner("Searching medical documents..."):
                results = retrieve_medical_context(question)

            if not results:
                st.info("No relevant passages were found.")

            for item in results:
                st.markdown(
                    f"**Source:** {item['source']} "
                    f"— page {item['page']}"
                )
                st.write(item["text"])
                st.divider()

        except Exception as exc:
            st.error(
                f"Search failed: {type(exc).__name__}: {exc}"
            )
