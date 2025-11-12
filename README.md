# Agentic Voice Assistant

> Voice-to-Voice AI Assistant for Product Discovery using LangGraph, MCP and RAG.

---

## Overview

This project builds a **voice-driven, multi-agent AI assistant** that helps users find and compare e-commerce products naturally.
It uses:

* **ASR (Automatic Speech Recognition)** to convert speech to text
* **LangGraph agents** for planning and reasoning
* **RAG (Retrieval-Augmented Generation)** to fetch product info from the Amazon 2020 dataset
* **MCP tools** for unified access to local and live sources
* **TTS (Text-to-Speech)** to reply back in natural voice

---

## Project Structure

```
agentic-voice-assistant/
├── main.py                     # Entry point
├── src/
│   ├── asr/                    # Speech-to-text logic
│   ├── tts/                    # Text-to-speech synthesis
│   ├── agents/                 # LangGraph agents (router, planner etc.)
│   ├── rag/                    # Private dataset indexing and retrieval
│   ├── mcp/                    # MCP server + tool schemas
│   └── ui/                     # Streamlit / frontend app
├── data/                       # Amazon dataset slices or indexes
├── notebooks/                  # Data prep & EDA
├── pyproject.toml              # Project and dev config
├── uv.lock                     # Locked dependencies
├── .pre-commit-config.yaml     # Pre-commit hooks for linting/formatting
└── .env.example                # Sample environment config (copy to .env)
```

---

## Local Development

1. **Clone the repo:**

   ```bash
   git clone git@github.com:sohammandal/agentic-voice-assistant.git
   cd agentic-voice-assistant
   ```

2. **Install [uv](https://github.com/astral-sh/uv)**:
    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```

3. **Install dependencies and set up virtual environment**:

   ```bash
   uv sync
   source .venv/bin/activate
   ```

4. **Set up pre-commit (run once):**

   ```bash
   pre-commit install
   
   # if pre-commit is missing then run this first
   uv tool install pre-commit
   ```

5. **(Optional) Run pre-commit on all files:**

   ```bash
   pre-commit run --all-files
   ```

   > This installs Git hooks so checks run automatically on every commit - you only need to do this once.

6. **Run the app (starter):**

   ```bash
   python main.py
   ```

### What Happens If Hooks Fail?

* If a hook auto-fixes code (e.g. formatting with `ruff`), the commit will be blocked.
* Simply **`git add .` and commit again** - your files will now be fixed and pass.
* If errors remain (like lint violations), you must resolve them manually before committing.

After the initial setup, hooks run automatically on changed files during `git commit` - **no need to run `pre-commit` manually each time**.

---

## Tech Stack

| Layer        | Tool                                |
| ------------ | ----------------------------------- |
| Voice Input  | Whisper / OpenAI ASR                |
| Reasoning    | LangGraph (multi-agent flow)        |
| Retrieval    | RAG (Amazon Product Dataset 2020)   |
| Tool Access  | MCP protocol (web + private search) |
| Voice Output | OpenAI / ElevenLabs TTS             |
| Frontend     | Streamlit                           |
| Dev Tools    | `uv`, `ruff`, `pre-commit`          |
