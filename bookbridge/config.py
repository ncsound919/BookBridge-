"""BookBridge configuration."""

import os
from pathlib import Path

# Data directory
DATA_DIR = Path(os.environ.get("BOOKBRIDGE_DATA_DIR", Path.home() / ".bookbridge"))
DB_PATH = DATA_DIR / "bookbridge.db"
CACHE_DIR = DATA_DIR / "cache"
OFFLINE_CACHE_MAX_MB = int(os.environ.get("BOOKBRIDGE_CACHE_MAX_MB", "2048"))

# Server addresses
HTTP_HOST = os.environ.get("BOOKBRIDGE_HTTP_HOST", "127.0.0.1")
HTTP_PORT = int(os.environ.get("BOOKBRIDGE_HTTP_PORT", "8777"))
MCP_HOST = os.environ.get("BOOKBRIDGE_MCP_HOST", "127.0.0.1")
MCP_PORT = int(os.environ.get("BOOKBRIDGE_MCP_PORT", "8778"))

# Indexing
CHUNK_SIZE_TOKENS = int(os.environ.get("BOOKBRIDGE_CHUNK_SIZE", "800"))
CHUNK_OVERLAP_TOKENS = int(os.environ.get("BOOKBRIDGE_CHUNK_OVERLAP", "100"))
EMBEDDING_DIMENSIONS = 384

# Ensure dirs exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)
