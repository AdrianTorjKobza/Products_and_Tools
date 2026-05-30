# MCP servers natively use standard input/output (stdio) for communication via JSON-RPC 2.0.
# This module asynchronously spawns the child processes (like Node.js or Python environments), listens to their output streams without blocking FastAPI, and sends execution payloads.

import subprocess
import json
import shutil
import os
import asyncio
from typing import Dict, Any, Optional

class MCPServerInstance:
    def __init__(self, name: str, command: str, args: list, env: dict = None):
        self.name = name
        self.command = command
        self.args = args
        
        # Inherit host OS environment variables
        self.env = os.environ.copy()
        if env:
            self.env.update(env)
            
        self.process: Optional[subprocess.Popen] = None
        self._id_counter = 1

    async def start(self):
        """Spawns the MCP server using standard subprocess to bypass Windows asyncio bugs."""
        try:
            resolved_cmd = shutil.which(self.command)
            
            if not resolved_cmd:
                print(f"❌ Failed: Command '{self.command}' not found in system PATH.")
                return

            # Combine command and args into a single list for Popen
            full_cmd = [resolved_cmd] + self.args

            # Use standard Popen. text=True handles string encoding automatically.
            self.process = subprocess.Popen(
                full_cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                env=self.env,
                text=True,
                bufsize=1 # Line-buffered for fast JSON-RPC streaming
            )
            print(f"📡 MCP Server '{self.name}' started successfully. (Bypassed Asyncio)")
        except Exception as e:
            print(f"❌ Failed to start MCP server '{self.name}': {repr(e)}")

    async def call_tool(self, tool_name: str, arguments: dict) -> Dict[str, Any]:
        """Sends a JSON-RPC request and reads the response asynchronously."""
        # poll() returns None if the process is still running
        if not self.process or self.process.poll() is not None:
            return {"error": "Server is not running or crashed."}

        req_id = self._id_counter
        self._id_counter += 1

        payload = {
            "jsonrpc": "2.0",
            "method": "tools/call",
            "params": {
                "name": tool_name,
                "arguments": arguments
            },
            "id": req_id
        }

        # 1. Write payload to standard input
        req_str = json.dumps(payload) + "\n"
        self.process.stdin.write(req_str)
        self.process.stdin.flush()

        # 2. Read from standard output in a background thread so we don't block FastAPI
        try:
            line = await asyncio.wait_for(
                asyncio.to_thread(self.process.stdout.readline), 
                timeout=10.0
            )
            if not line:
                return {"error": "MCP Server returned an empty response."}
            return json.loads(line)
        except asyncio.TimeoutError:
            return {"error": "MCP Server response timeout."}
        except Exception as e:
            return {"error": f"Failed to parse MCP response: {str(e)}"}

    async def stop(self):
        if self.process:
            try:
                self.process.terminate()
                self.process.wait(timeout=2)
            except Exception:
                self.process.kill()

class MCPManager:
    def __init__(self):
        self.instances: Dict[str, MCPServerInstance] = {}

    async def start_server(self, name: str, config: dict):
        if name in self.instances:
            await self.instances[name].stop()
        
        instance = MCPServerInstance(name, config["command"], config["args"], config.get("env"))
        await instance.start()
        self.instances[name] = instance

    async def stop_all(self):
        for instance in self.instances.values():
            await instance.stop()

# Global tracking instance
mcp_manager = MCPManager()