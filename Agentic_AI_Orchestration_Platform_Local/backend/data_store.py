# Persisted configured MCP servers and Skills across server restarts.
# Simple JSON file database wrapped in safe read/write operations.

import json
import os
import sys
from typing import Dict, List, Any

DATA_FILE = "platform_config.json"

# 1. Dynamically resolve OS-specific executable for npx
NPX_CMD = "npx.cmd" if sys.platform == "win32" else "npx"

# 2. Dynamically resolve the absolute path to the workspace folder in the project root
WORKSPACE_DIR = os.path.abspath(os.path.join(os.getcwd(), "agent-workspace"))

# 3. Automatically create the directory if it doesn't exist
os.makedirs(WORKSPACE_DIR, exist_ok=True)

DEFAULT_DATA = {
    "mcp_servers": {
        "filesystem": {
            "command": NPX_CMD,
            "args": ["-y", "@modelcontextprotocol/server-filesystem", WORKSPACE_DIR],
            "env": {},
            "enabled": True
        }
    },
    "skills": {
        "calculator": {
            "code": "def execute(a, b):\n    return a + b",
            "description": "Adds two numbers together when standard LLM math fails.",
            "enabled": True
        }
    }
}

def load_store() -> Dict[str, Any]:
    if not os.path.exists(DATA_FILE):
        with open(DATA_FILE, "w") as f:
            json.dump(DEFAULT_DATA, f, indent=4)
        return DEFAULT_DATA
    with open(DATA_FILE, "r") as f:
        return json.load(f)

def save_store(data: Dict[str, Any]):
    with open(DATA_FILE, "w") as f:
        json.dump(data, f, indent=4)

def get_mcp_servers() -> Dict[str, Any]:
    return load_store().get("mcp_servers", {})

def save_mcp_server(name: str, config: Dict[str, Any]):
    store = load_store()
    store["mcp_servers"][name] = config
    save_store(store)

def get_skills() -> Dict[str, Any]:
    return load_store().get("skills", {})

def save_skill(name: str, config: Dict[str, Any]):
    store = load_store()
    store["skills"][name] = config
    save_store(store)

def delete_mcp_server(name: str):
    store = load_store()
    if name in store["mcp_servers"]:
        del store["mcp_servers"][name]
        save_store(store)

def delete_skill(name: str):
    store = load_store()
    if name in store["skills"]:
        del store["skills"][name]
        save_store(store)