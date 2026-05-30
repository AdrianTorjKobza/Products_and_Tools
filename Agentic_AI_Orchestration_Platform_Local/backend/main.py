# This is the primary API runtime. It hosts endpoints for configuring assets and managing the Human-in-the-Loop state gate.
# If the agent yields a tool_call, the session pauses, caches the pending task state, and flags the execution loop as AWAITING_APPROVAL.

import sys
import asyncio
import contextlib
import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import json

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

from backend.data_store import get_mcp_servers, save_mcp_server, get_skills, save_skill, delete_mcp_server, delete_skill
from backend.mcp_manager import mcp_manager
from backend.agent import run_agent_turn

@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    servers = get_mcp_servers()
    for name, config in servers.items():
        if config.get("enabled", True):
            await mcp_manager.start_server(name, config)
    yield
    await mcp_manager.stop_all()

app = FastAPI(title="Protos Agentic Hub", lifespan=lifespan)

SESSION_STATE = {
    "messages": [],
    "status": "idle",
    "pending_tool": None,
    "model_name": "llama3"
}

class ChatInput(BaseModel):
    message: str
    model_name: str

class ServerInput(BaseModel):
    name: str
    command: str
    args: List[str]
    env: Optional[Dict[str, str]] = {}

class SkillInput(BaseModel):
    name: str
    code: str
    description: str

@app.get("/state")
def get_state():
    return SESSION_STATE

# --- NEW: Clear Chat Endpoint ---
@app.post("/chat/clear")
def clear_chat():
    SESSION_STATE["messages"] = []
    SESSION_STATE["status"] = "idle"
    SESSION_STATE["pending_tool"] = None
    return {"status": "success"}
# --------------------------------

@app.get("/models")
async def get_ollama_models():
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get("http://localhost:11434/api/tags", timeout=5.0)
            return resp.json().get("models", [])
    except Exception:
        return []

@app.post("/chat")
async def chat_endpoint(payload: ChatInput):
    if SESSION_STATE["status"] == "awaiting_approval":
        raise HTTPException(status_code=400, detail="An operation is waiting for authorization.")

    SESSION_STATE["messages"].append({"role": "user", "content": payload.message})
    SESSION_STATE["status"] = "running"
    SESSION_STATE["model_name"] = payload.model_name
    
    agent_output = await run_agent_turn(SESSION_STATE["messages"], SESSION_STATE["model_name"])
    
    if agent_output.get("status") == "tool_call":
        SESSION_STATE["status"] = "awaiting_approval"
        SESSION_STATE["pending_tool"] = agent_output
        return {"status": "awaiting_approval", "action_required": agent_output}
    
    reply = agent_output.get("reply", "No distinct answer yielded.")
    SESSION_STATE["messages"].append({"role": "assistant", "content": reply})
    SESSION_STATE["status"] = "idle"
    return {"status": "idle", "reply": reply}

@app.post("/chat/approve")
async def approve_tool():
    if SESSION_STATE["status"] != "awaiting_approval" or not SESSION_STATE["pending_tool"]:
        raise HTTPException(status_code=400, detail="No action is awaiting approval.")
    
    tool = SESSION_STATE["pending_tool"]
    SESSION_STATE["status"] = "running"
    SESSION_STATE["pending_tool"] = None

    result = {}
    if tool["tool_type"] == "mcp":
        server_name = tool["target"]
        if server_name in mcp_manager.instances:
            result = await mcp_manager.instances[server_name].call_tool(tool["method"], tool["arguments"])
        else:
            result = {"error": f"MCP server '{server_name}' is currently unavailable."}
    elif tool["tool_type"] == "skill":
        skills = get_skills()
        if tool["target"] in skills:
            try:
                local_vars = {}
                exec(skills[tool["target"]]["code"], {}, local_vars)
                func = local_vars.get("execute")
                if func:
                    result = {"result": func(**tool["arguments"])}
                else:
                    result = {"error": "Skills execution requires an explicit 'execute' entrypoint function."}
            except Exception as e:
                result = {"error": f"Skill exception error: {str(e)}"}

    SESSION_STATE["messages"].append({"role": "system", "content": f"Tool Execution Result: {json.dumps(result)}"})
    
    next_turn = await run_agent_turn(SESSION_STATE["messages"], SESSION_STATE["model_name"])
    if next_turn.get("status") == "tool_call":
        SESSION_STATE["status"] = "awaiting_approval"
        SESSION_STATE["pending_tool"] = next_turn
        return {"status": "awaiting_approval", "action_required": next_turn}

    reply = next_turn.get("reply", "Finished processing step.")
    SESSION_STATE["messages"].append({"role": "assistant", "content": reply})
    SESSION_STATE["status"] = "idle"
    return {"status": "idle", "reply": reply}

@app.post("/chat/reject")
def reject_tool():
    if SESSION_STATE["status"] != "awaiting_approval":
        raise HTTPException(status_code=400, detail="No actions require processing.")
    
    SESSION_STATE["pending_tool"] = None
    SESSION_STATE["status"] = "idle"
    SESSION_STATE["messages"].append({"role": "system", "content": "Action rejected by human operator."})
    return {"status": "idle", "msg": "Tool execution was successfully denied."}

@app.get("/config")
def get_all_config():
    return {"mcp_servers": get_mcp_servers(), "skills": get_skills()}

@app.post("/config/mcp")
async def add_mcp(config: ServerInput):
    srv_dict = {"command": config.command, "args": config.args, "env": config.env, "enabled": True}
    save_mcp_server(config.name, srv_dict)
    await mcp_manager.start_server(config.name, srv_dict)
    return {"status": "success"}

@app.delete("/config/mcp/{name}")
async def remove_mcp(name: str):
    delete_mcp_server(name)
    if name in mcp_manager.instances:
        await mcp_manager.instances[name].stop()
        del mcp_manager.instances[name]
    return {"status": "success"}

@app.post("/config/skill")
def add_skill(config: SkillInput):
    skill_dict = {"code": config.code, "description": config.description, "enabled": True}
    save_skill(config.name, skill_dict)
    return {"status": "success"}

@app.delete("/config/skill/{name}")
def remove_skill(name: str):
    delete_skill(name)
    return {"status": "success"}