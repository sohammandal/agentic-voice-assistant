.PHONY: run server app rag-start rag-stop

rag-start:
	uv run python -m src.rag.run_service start

rag-stop:
	uv run python -m src.rag.run_service stop

server:
	uv run python -m src.mcp.mcp_server

app:
	uv run python main.py

run:
	@set -e; \
	trap 'uv run python -m src.rag.run_service stop || true; \
	      kill $$server_pid 2>/dev/null || true' EXIT INT TERM; \
	uv run python -m src.rag.run_service start; \
	uv run python -m src.mcp.mcp_server & \
	server_pid=$$!; \
	uv run python main.py
