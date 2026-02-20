"""BookBridge entry point.

Run the HTTP daemon (port 8777) and the MCP server (port 8778)
as concurrent services in a single process.

Usage:
  python main.py                 # starts both servers
  python main.py --http-only     # HTTP only
  python main.py --mcp-only      # MCP only
"""

from __future__ import annotations

import argparse
import threading

import uvicorn

from bookbridge.config import HTTP_HOST, HTTP_PORT, MCP_HOST, MCP_PORT
from bookbridge.database import DB_PATH, init_db
from bookbridge.server import app as http_app
from bookbridge.mcp_server import mcp_app


def _run_http():
    uvicorn.run(http_app, host=HTTP_HOST, port=HTTP_PORT, log_level="info")


def _run_mcp():
    uvicorn.run(mcp_app, host=MCP_HOST, port=MCP_PORT, log_level="info")


def main():
    parser = argparse.ArgumentParser(description="BookBridge daemon")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--http-only", action="store_true", help="Run HTTP server only")
    group.add_argument("--mcp-only", action="store_true", help="Run MCP server only")
    args = parser.parse_args()

    # Ensure DB schema is up to date
    init_db(DB_PATH)

    if args.http_only:
        _run_http()
    elif args.mcp_only:
        _run_mcp()
    else:
        # Run both concurrently
        http_thread = threading.Thread(target=_run_http, daemon=True)
        http_thread.start()
        # MCP server runs in the main thread
        _run_mcp()


if __name__ == "__main__":
    main()
