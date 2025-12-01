"""
main.py

Entry point for the Agentic Voice-to-Voice Product Discovery System.

This file manages:
- ASR input (placeholder)
- Sending text to graph workflow
- Maintaining persistent conversation state across turns
- Sending final responses to TTS/UI layer (placeholders included)
"""

# Load environment variables from a local .env file if python-dotenv is available.
# This ensures keys like GROQ_API_KEY are present in os.environ before modules
# (which read from os.environ at import time) are imported.
try:
    from dotenv import find_dotenv, load_dotenv

    # find_dotenv() returns the path to a .env file in the repo if present
    load_dotenv(find_dotenv() or ".env")
except Exception:
    # If python-dotenv is not installed, nothing to do here — instruct user to
    # set env vars in their shell instead.
    pass

from src.orchestration.graph import initialize_state, run_graph


# -------------------------------------------------------------------------
# OPTIONAL: Placeholder ASR function (replace later)
# -------------------------------------------------------------------------
def asr_stub():
    """
    Simulate ASR.
    Replace this with your real speech-to-text pipeline.
    """
    user_text = input("🎤 Say something (type your text): ")
    return user_text


# -------------------------------------------------------------------------
# OPTIONAL: Placeholder TTS function (replace later)
# -------------------------------------------------------------------------
def tts_stub(response_text: str):
    """
    Simulate TTS.
    Replace this with your real text-to-speech pipeline.
    """
    print("\n🔊 TTS Output:")
    print(response_text)
    print("\n" + "=" * 60 + "\n")


# -------------------------------------------------------------------------
# CONSOLE LOOP — Your main runtime
# -------------------------------------------------------------------------
def main():
    """
    Main loop:
    - Maintains persistent conversation state
    - Reads user text (ASR stub for now)
    - Runs LangGraph workflow
    - Outputs final text to "UI" + TTS stub
    """

    print("\n🚀 Agentic Voice Assistant Initialized")
    print("Type your query. Type 'exit' to quit.\n")

    # Initialize persistent conversation state
    state = initialize_state()

    while True:
        # Step 1: Get user text (ASR stub)
        user_text = asr_stub()

        if user_text.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break

        # Step 2: Run graph
        updated_state = run_graph(user_text, state)

        # Step 3: Extract final response text
        final_text = updated_state.get("response_text", "(no response generated)")

        # Step 4: Output to UI and TTS
        print("\n💬 Assistant:")
        print(final_text)
        print("\n" + "=" * 60 + "\n")

        tts_stub(final_text)

        # Step 5: Keep state for next turn
        state = updated_state


# -------------------------------------------------------------------------
# ENTRYPOINT
# -------------------------------------------------------------------------
if __name__ == "__main__":
    main()
