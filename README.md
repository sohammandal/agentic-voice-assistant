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
│   ├── orchestration           # LangGraph agents (router, planner etc.)
│   ├── llm                     # LLM API Call Utility
│   ├── rag/                    # Private dataset indexing and retrieval
│   ├── mcp/                    # MCP server + tool schemas
│   └── ui/                     # Streamlit / frontend app
├── data/                       # Amazon dataset slices or indexes
├── notebooks/                  # Data prep & EDA
├── vector_store/               # Local directory with Chrome DB (gitignored)
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

   > This installs Git hooks so checks run automatically on every commit - you only need to do this once.

5. **(Optional) Run pre-commit on all files:**

   ```bash
   pre-commit run --all-files
   ```

6. **Run the app:**

   From the project root, run these two commands in separate terminals:

   ```bash
   python -m src.rag.run_service start
   ```

   ```bash
   make run
   ```

7. **Try the assistant:**

   Ask a question like:
   * _"Recommend window privacy films under $15"_
   * _"Recommend stainless steel hose clamps under $25"_


### What Happens If Hooks Fail?

* If a hook auto-fixes code (e.g. formatting with `ruff`), the commit will be blocked.
* Simply **`git add .` and commit again** - your files will now be fixed and pass.
* If errors remain (like lint violations), you must resolve them manually before committing.

After the initial setup, hooks run automatically on changed files during `git commit` - **no need to run `pre-commit` manually each time**.

---

### Adding a New Package/Dependency

If you need to add a package to the project:

```bash
# 1. Add the package (updates pyproject.toml and uv.lock)
uv add <package-name>

# example:
# uv add scikit-learn
```

`uv` will automatically install the package and make sure all dependencies stay compatible.

Finally, commit the updated files:

```bash
git add pyproject.toml uv.lock
git commit -m "Add <package-name>"
```

Everyone else should pull the changes and run:

```bash
uv sync
```

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
