.PHONY: run server app rag-start rag-stop

PYTHON = uv run python
STREAMLIT = uv run streamlit

rag-start:
	$(PYTHON) -m src.rag.run_service start

rag-stop:
	$(PYTHON) -m src.rag.run_service stop

server:
	$(PYTHON) -m src.mcp.mcp_server

# Streamlit app only (assumes servers already running)
app:
	$(STREAMLIT) run app.py

# Full stack: RAG + MCP + Streamlit
run:
	@set -e; \
	trap '$(PYTHON) -m src.rag.run_service stop || true; \
	      kill $$server_pid 2>/dev/null || true' EXIT INT TERM; \
	$(PYTHON) -m src.rag.run_service start; \
	$(PYTHON) -m src.mcp.mcp_server & \
	server_pid=$$!; \
	$(STREAMLIT) run app.py
