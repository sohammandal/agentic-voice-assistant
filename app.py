"""
app.py - streamlit app for Agentic Voice Assistant
"""

# Load environment variables
from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv() or ".env")

# Import necessary libraries
import io
import os
import re
import tempfile

import soundfile as sf
import streamlit as st
from num2words import num2words

from src.asr.asr import ASRProcessor
from src.orchestration.graph import initialize_state, run_graph
from src.tts.tts import TTSProcessor

# Configure the page
st.set_page_config(
    page_title="Agentic Voice Assistant",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---- Session State Configuration ----

if "conversation_state" not in st.session_state:
    st.session_state.conversation_state = initialize_state()

if "last_response" not in st.session_state:
    st.session_state.last_response = None

if "audio_response" not in st.session_state:
    st.session_state.audio_response = None

if "transcribed_text" not in st.session_state:
    st.session_state.transcribed_text = None

if "audio_widget_key" not in st.session_state:
    st.session_state.audio_widget_key = 0

if "last_query" not in st.session_state:
    st.session_state.last_query = None

if "voice_gender" not in st.session_state:
    st.session_state.voice_gender = "male"  # Default voice

# ---- ASR and TTS Model Initialization ----

if "asr_processor" not in st.session_state:
    with st.spinner("Loading ASR model..."):
        st.session_state.asr_processor = ASRProcessor(model_name="small")

if "tts_processor" not in st.session_state:
    with st.spinner("Loading TTS model..."):
        st.session_state.tts_processor = TTSProcessor(
            gender=st.session_state.voice_gender
        )


# ---- Helper Function Library ----
def preprocess_text_for_tts(text):
    """Convert numbers to words for TTS and remove markdown asterisks."""
    text = text.replace("*", "")

    def replace_dollar(match):
        return f"{num2words(round(float(match.group(1))))} dollars"

    text = re.sub(r"\$(\d+\.?\d*)", replace_dollar, text)

    def replace_number(match):
        return num2words(round(float(match.group(0))))

    text = re.sub(r"\b(\d+\.?\d*)\b", replace_number, text)

    return re.sub(r"\s+", " ", text).strip()


def fix_section_headers(text: str) -> str:
    """Clean up headers and rebuild item lines to ensure clean formatting."""
    # Clean Headers
    text = re.sub(
        r"(?m)^.*Catalog Items \(Private Amazon-2020 Dataset\).*?:\s*(.*)",
        r"### Catalog Items (Private Amazon-2020 Dataset):\n\1",
        text,
    )
    text = re.sub(
        r"(?m)^.*Live Web Items.*?:?\s*(.*)", r"### Live Web Items:\n\1", text
    )

    # Rebuild Item Lines
    lines = text.split("\n")
    result_lines = []
    in_catalog = False
    in_web = False

    for line in lines:
        stripped = line.strip()

        # Track sections
        if stripped.startswith("### Catalog Items"):
            in_catalog = True
            in_web = False
            result_lines.append(line)
            continue
        elif stripped.startswith("### Live Web Items"):
            in_catalog = False
            in_web = True
            result_lines.append(line)
            continue
        elif stripped.startswith("###"):
            in_catalog = False
            in_web = False
            result_lines.append(line)
            continue

        # Identify Product Titles
        is_title = (
            stripped.startswith("**")
            and stripped.endswith("**")
            and not stripped.startswith("• ")
            and len(stripped) > 5
        )

        if is_title and (in_catalog or in_web):
            # Remove existing bolds, emojis, stars
            clean = (
                stripped.replace("*", "").replace("🛒", "").replace("🌐", "").strip()
            )
            # Rebuild with single emoji and single bold
            emoji = "🛒" if in_catalog else "🌐"
            result_lines.append(f"**{emoji} {clean}**")
        else:
            result_lines.append(line)

    return "\n".join(result_lines)


def process_audio_file(audio_file):
    """Process audio with robust temp file cleanup."""
    tmp_path = None
    try:
        # Save upload to temp file for ASR
        data = audio_file.read() if hasattr(audio_file, "read") else audio_file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            tmp_file.write(data)
            tmp_path = tmp_file.name

        # Process
        result = st.session_state.asr_processor.encode_audio(tmp_path)
        return result.get("text", "")

    except Exception as e:
        st.error(f"Error processing audio: {str(e)}")
        return None

    finally:
        # Delete the temporary file
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


def generate_tts_audio(text):
    """Generate TTS audio via TTSProcessor from custom TTS module."""
    try:
        processed_text = preprocess_text_for_tts(text)
        waveform = st.session_state.tts_processor.decode_audio(processed_text)

        # Write to memory buffer instead of disk
        virtual_file = io.BytesIO()
        sf.write(virtual_file, waveform.cpu().numpy(), 16000, format="WAV")
        virtual_file.seek(0)

        return virtual_file

    except Exception as e:
        st.error(f"Error generating TTS: {str(e)}")
        return None


def extract_first_paragraph(text):
    """Get the first paragraph from the response text."""
    parts = text.split("\n\n", 1)
    return parts[0] if parts else text


def get_remaining_content(text):
    """Get remaining content after the first paragraph, fixing headers."""
    parts = text.split("\n\n", 1)
    if len(parts) > 1:
        return fix_section_headers(parts[1])
    return ""


# ---- Page Layout Configuration ----

# Create buttons
# Custom CSS for button colors
st.markdown(
    """
<style>
    /* Primary (selected) button - blue */
    button[kind="primary"] { 
        background-color: #0068C9 !important; 
        border-color: #0068C9 !important; 
        color: white !important;
    }
    button[kind="primary"]:hover { 
        background-color: #0054A3 !important; 
        border-color: #0054A3 !important; 
    }
    
    /* Secondary (unselected) button - sidebar gray */
    button[kind="secondary"] { 
        background-color: #f0f2f6 !important; 
        border-color: #d3d3d3 !important; 
        color: #262730 !important; 
    }
    button[kind="secondary"]:hover { 
        background-color: #e0e2e6 !important; 
        border-color: #b8bac0 !important; 
    }
</style>
""",
    unsafe_allow_html=True,
)

# Add title and sidebar
st.title("Agentic Voice Assistant")
st.subheader("Voice-to-Voice AI for Product Discovery")

with st.sidebar:
    st.header("Voice Settings")
    st.write("**Select Voice Gender:**")

    col1, col2 = st.columns(2)

    with col1:
        if st.button(
            "🎙️ Male",
            use_container_width=True,
            type="primary" if st.session_state.voice_gender == "male" else "secondary",
        ):
            if st.session_state.voice_gender != "male":
                st.session_state.voice_gender = "male"
                if "tts_processor" in st.session_state:
                    st.session_state.tts_processor.change_voice("male")
                st.success("Voice changed to Male")
                st.rerun()

    with col2:
        if st.button(
            "🎙️ Female",
            use_container_width=True,
            type="primary"
            if st.session_state.voice_gender == "female"
            else "secondary",
        ):
            if st.session_state.voice_gender != "female":
                st.session_state.voice_gender = "female"
                if "tts_processor" in st.session_state:
                    st.session_state.tts_processor.change_voice("female")
                st.success("Voice changed to Female")
                st.rerun()

    st.caption(f"Current: {st.session_state.voice_gender.title()} voice")

    st.divider()
    st.header("Input Options")
    input_method = st.radio(
        "Choose input method:",
        ["Record Audio", "Type Text", "Upload Audio"],
        key="input_method",
    )
    st.divider()
    st.header("About")
    st.info("Uses Voice input, Multi-agent reasoning, and RAG retrieval.")

    if st.button("🔄 Reset Conversation"):
        st.session_state.conversation_state = initialize_state()
        st.session_state.last_response = None
        st.session_state.audio_response = None
        st.session_state.transcribed_text = None
        st.session_state.audio_widget_key += 1
        st.session_state.last_query = None
        st.session_state.voice_gender = "male"  # Reset to default

        # Clear models to force reload
        if "asr_processor" in st.session_state:
            del st.session_state.asr_processor
        if "tts_processor" in st.session_state:
            del st.session_state.tts_processor

        st.success("Reset complete. Models will reload.")
        st.rerun()

st.subheader(f"Selected: {input_method}")
user_query = None

if st.session_state.last_response and st.session_state.last_query:
    st.info(f"**Your Question:** {st.session_state.last_query}")

# ---- Input Logic ----

if input_method == "Type Text":
    user_query = st.text_input("Enter your question:", key="text_input")
    if user_query:
        st.write(f"You entered: **{user_query}**")

elif input_method in ["Record Audio", "Upload Audio"]:
    if input_method == "Record Audio":
        st.write("Click to record:")
        audio_file = st.audio_input(
            "Record", key=f"audio_{st.session_state.audio_widget_key}"
        )
    else:
        st.write("Upload audio:")
        # FIXED: Added "m4a" and "flac" back to the allowed file types
        audio_file = st.file_uploader(
            "Upload",
            type=["wav", "mp3", "m4a", "flac"],
            key=f"upload_{st.session_state.audio_widget_key}",
        )

    if audio_file:
        st.audio(audio_file)
        c1, c2 = st.columns([3, 1])

        if c1.button("Transcribe Audio", type="primary", use_container_width=True):
            with st.spinner("Transcribing..."):
                text = process_audio_file(audio_file)
                if text:
                    st.session_state.transcribed_text = text
                    st.rerun()

        if c2.button("Clear/Re-record", type="secondary", use_container_width=True):
            st.session_state.transcribed_text = None
            st.session_state.audio_widget_key += 1
            st.rerun()

    if st.session_state.transcribed_text:
        st.success(f"Transcribed: {st.session_state.transcribed_text}")
        user_query = st.session_state.transcribed_text

# ---- Query Processing Logic ----

if user_query:
    if st.button("Process Query", type="primary", use_container_width=True):
        with st.spinner("Processing query..."):
            try:
                updated_state = run_graph(
                    user_query, st.session_state.conversation_state
                )
                st.session_state.conversation_state = updated_state

                resp_text = updated_state.get("response_text", "")
                first_para = extract_first_paragraph(resp_text)

                # Generate Audio in Memory
                audio_bytes = generate_tts_audio(first_para)

                st.session_state.last_response = {
                    "text": resp_text,
                    "step_log": updated_state.get("step_log", []),
                }
                st.session_state.audio_response = audio_bytes
                st.session_state.last_query = user_query
                st.session_state.transcribed_text = None

                st.success("Processed!")
                st.rerun()

            except Exception as e:
                st.error(f"Error: {e}")

# ---- Answering Logic ----

if st.session_state.last_response:
    st.divider()
    st.header("Assistant Response")

    if st.session_state.audio_response:
        st.audio(st.session_state.audio_response, format="audio/wav")

    remaining = get_remaining_content(st.session_state.last_response["text"])
    if remaining:
        st.divider()
        st.markdown(remaining)

    st.divider()
    with st.expander("Agent Step Log"):
        for i, step in enumerate(st.session_state.last_response["step_log"], 1):
            st.markdown(f"**{i}.** {step}")

st.divider()
