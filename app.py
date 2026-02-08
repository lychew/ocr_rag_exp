#!/usr/bin/env python3
"""
Streamlit Chatbot UI for RAG Pipeline.

Features:
- Chat interface for Q&A about the book
- Select any OCR + Chunking combination
- Optional VLM enhancement for figure understanding
- Display supporting chunks and page images

Usage:
    streamlit run app.py
"""

import streamlit as st
from pathlib import Path
import sys
import base64

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.pipeline import RAGPipeline
from src.generation.prompt_templates import PromptStrategy


# Configuration
OCR_MODELS = ["tesseract", "easyocr", "paddleocr"]
CHUNKING_STRATEGIES = ["page", "parent_child", "semantic"]
PROMPT_STRATEGIES = ["basic", "cot", "few_shot"]
PAGES_DIR = Path("data/pages")


def get_page_image_base64(page_number: int) -> str | None:
    """Get base64 encoded page image for display."""
    image_path = PAGES_DIR / f"page_{page_number:04d}.png"
    if image_path.exists():
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    return None


def check_collection_exists(ocr: str, chunking: str) -> bool:
    """Check if a collection has been indexed."""
    try:
        pipeline = RAGPipeline(ocr_name=ocr, chunking_name=chunking)
        return pipeline.vectorstore.count() > 0
    except Exception:
        return False


def get_vlm_figure_description(page_number: int, question: str) -> str | None:
    """Use VLM to describe figures on a page related to the question."""
    image_path = PAGES_DIR / f"page_{page_number:04d}.png"
    if not image_path.exists():
        return None

    try:
        from src.ocr.vlm_extractor import VLMExtractor
        vlm = VLMExtractor(model_name="gpt-4o-mini")  # Use mini for cost efficiency

        # Custom prompt for figure analysis
        import base64
        from openai import OpenAI
        import os

        with open(image_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode("utf-8")

        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are analyzing a scanned book page. "
                        "Look for any diagrams, figures, charts, or illustrations. "
                        "If found, describe them in detail. "
                        "If no figures are present, respond with 'NO_FIGURES'. "
                        "Focus on aspects relevant to the user's question."
                    ),
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{image_data}"},
                        },
                        {
                            "type": "text",
                            "text": f"Question context: {question}\n\nDescribe any figures or diagrams on this page that might be relevant.",
                        },
                    ],
                },
            ],
            max_tokens=500,
            temperature=0.0,
        )

        result = response.choices[0].message.content.strip()
        if "NO_FIGURES" in result:
            return None
        return result
    except Exception as e:
        return f"VLM error: {str(e)}"


# Page config
st.set_page_config(
    page_title="Public Health RAG Chatbot",
    page_icon="📚",
    layout="wide",
)

# Title
st.title("📚 Principles of Public Health - RAG Chatbot")
st.markdown("Ask questions about the book and get grounded answers with citations.")

# Sidebar configuration
with st.sidebar:
    st.header("⚙️ Configuration")

    # OCR selection
    selected_ocr = st.selectbox(
        "OCR Model",
        OCR_MODELS,
        index=0,  # Default: tesseract
        help="Select the OCR model used for text extraction"
    )

    # Chunking selection
    selected_chunking = st.selectbox(
        "Chunking Strategy",
        CHUNKING_STRATEGIES,
        index=2,  # Default: semantic
        help="Select how the text was split into chunks"
    )

    # Prompt strategy
    selected_prompt = st.selectbox(
        "Prompt Strategy",
        PROMPT_STRATEGIES,
        index=0,  # Default: basic
        help="basic: simple, cot: chain-of-thought, few_shot: with examples"
    )

    # VLM enhancement toggle
    vlm_enabled = st.toggle(
        "🔮 VLM Figure Enhancement",
        value=False,
        help="Use GPT-4o to analyze figures/diagrams on retrieved pages (adds latency & cost)"
    )

    st.divider()

    # Check if collection exists
    collection_name = f"{selected_ocr}__{selected_chunking}"
    collection_exists = check_collection_exists(selected_ocr, selected_chunking)

    if collection_exists:
        st.success(f"✅ Collection ready: `{collection_name}`")
    else:
        st.error(f"❌ Collection not found: `{collection_name}`")
        st.code(f"uv run scripts/setup.py --ocr {selected_ocr} --chunking {selected_chunking}")

    st.divider()

    # Example questions
    st.subheader("📝 Example Questions")
    example_questions = [
        "What are the main ways to fight disease germs?",
        "How does pure air affect health?",
        "What is the relationship between cleanliness and disease prevention?",
    ]
    for q in example_questions:
        if st.button(q, key=f"example_{q[:20]}", use_container_width=True):
            st.session_state.example_question = q


# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "chunks" in message:
            with st.expander("📄 Supporting Chunks"):
                for chunk in message["chunks"]:
                    st.markdown(f"**{chunk['chunk_id']}** (Page {chunk['page']})")
        if "figure_analysis" in message and message["figure_analysis"]:
            with st.expander("🖼️ Figure Analysis (VLM)"):
                st.markdown(message["figure_analysis"])

# Handle example question button
if "example_question" in st.session_state:
    prompt = st.session_state.example_question
    del st.session_state.example_question
else:
    prompt = st.chat_input("Ask a question about the book...")

# Process user input
if prompt:
    # Check collection exists
    if not collection_exists:
        st.error("Please set up the collection first. See sidebar for instructions.")
    else:
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # Create pipeline
                    pipeline = RAGPipeline(
                        ocr_name=selected_ocr,
                        chunking_name=selected_chunking,
                        prompt_strategy=selected_prompt,
                    )

                    # Get answer
                    response, retrieved_chunks = pipeline.answer_question(prompt)

                    # VLM enhancement for figures
                    figure_analysis = None
                    if vlm_enabled and retrieved_chunks:
                        with st.spinner("Analyzing figures with VLM..."):
                            # Get unique pages from retrieved chunks
                            pages = set(rc.chunk.page_number for rc in retrieved_chunks)
                            figure_descriptions = []

                            for page_num in pages:
                                desc = get_vlm_figure_description(page_num, prompt)
                                if desc:
                                    figure_descriptions.append(f"**Page {page_num}:** {desc}")

                            if figure_descriptions:
                                figure_analysis = "\n\n".join(figure_descriptions)

                    # Display answer
                    st.markdown(response.answer)

                    # Supporting chunks
                    chunks_data = [
                        {"chunk_id": sc.chunk_id, "page": sc.page}
                        for sc in response.supporting_chunks
                    ]

                    with st.expander("📄 Supporting Chunks", expanded=False):
                        for i, rc in enumerate(retrieved_chunks):
                            col1, col2 = st.columns([3, 1])
                            with col1:
                                st.markdown(f"**{rc.chunk.chunk_id}** (Page {rc.chunk.page_number}, Score: {rc.score:.3f})")
                                st.text(rc.chunk.text[:500] + "..." if len(rc.chunk.text) > 500 else rc.chunk.text)
                            with col2:
                                # Show page thumbnail
                                img_b64 = get_page_image_base64(rc.chunk.page_number)
                                if img_b64:
                                    st.image(f"data:image/png;base64,{img_b64}", width=150)
                            st.divider()

                    # VLM figure analysis
                    if figure_analysis:
                        with st.expander("🖼️ Figure Analysis (VLM)", expanded=True):
                            st.markdown(figure_analysis)

                    # Save to history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response.answer,
                        "chunks": chunks_data,
                        "figure_analysis": figure_analysis,
                    })

                except Exception as e:
                    st.error(f"Error: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())

# Footer
st.sidebar.divider()
st.sidebar.markdown(
    """
    ---
    **RAG Pipeline** for *Principles of Public Health*

    Built with Streamlit + ChromaDB + GPT-4o
    """
)
