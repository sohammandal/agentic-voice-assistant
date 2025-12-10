.PHONY: run server app

server:
	uv run python -m src.mcp.mcp_server

app:
	uv run python main.py

run:
	@set -e; \
	uv run python -m src.mcp.mcp_server & \
	server_pid=$$!; \
	trap 'kill $$server_pid 2>/dev/null || true' EXIT INT TERM; \
	uv run python main.py
