# This module communicates with the local Ollama instance.

import httpx
import json
from typing import Dict, Any, List

# 1. Import the datastore to read active tools
from backend.data_store import get_mcp_servers, get_skills

OLLAMA_URL = "http://localhost:11434/api/chat"

def build_system_prompt() -> str:
    servers = get_mcp_servers()
    skills = get_skills()
    
    # We must explicitly tell the LLM that if it's not in this list, it DOES NOT EXIST
    tool_list = "\n".join([f"- MCP: {name}" for name in servers] + [f"- Skill: {name}" for name in skills])
    
    return f"""You are a helpful AI assistant.

### AVAILABLE TOOLS:
{tool_list if tool_list else "None"}

### DECISION LOGIC:
1. If the user says "hi", "hello", or asks a general knowledge question, DO NOT use a tool. Reply directly.
2. ONLY invoke a tool if you specifically need external data, file access, or calculations.
3. NEVER hallucinate tool targets. You may ONLY target the exact tool names listed in the AVAILABLE TOOLS section above.

### RESPONSE SCHEMAS:

Schema 1: Direct Reply (Use for greetings and general chat)
{{
    "status": "complete",
    "reply": "Your markdown formatted message here"
}}

Schema 2: Tool Invocation (Use ONLY when necessary)
{{
    "status": "tool_call",
    "tool_type": "mcp" or "skill",
    "target": "EXACT_NAME_FROM_AVAILABLE_TOOLS",
    "method": "specific_action_to_take",
    "arguments": {{ "key": "value" }},
    "thought": "Why you are running this specific tool"
}}
"""

async def run_agent_turn(messages: List[Dict[str, str]], model_name: str) -> Dict[str, Any]:
    dynamic_prompt = build_system_prompt()
    formatted_messages = [{"role": "system", "content": dynamic_prompt}] + messages
    
    payload = {"model": model_name, "messages": formatted_messages, "stream": False}

    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(OLLAMA_URL, json=payload, timeout=60.0)
            res_json = response.json()
            raw_content = res_json.get("message", {}).get("content", "").strip()
            
            # Attempt 1: Standard JSON parse
            try:
                return json.loads(raw_content)
            except:
                # Attempt 2: If the model gave plain text, WRAP IT!
                # This stops the "Model failed to return valid JSON" errors.
                return {
                    "status": "complete", 
                    "reply": raw_content
                }
        except Exception as e:
            return {"status": "complete", "reply": f"System Error: {str(e)}"}