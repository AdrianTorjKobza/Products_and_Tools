import streamlit as st
import requests

BACKEND_URL = "http://localhost:8000"

st.set_page_config(page_title="Protos Agentic Workspace", layout="wide")

# --- HEADER & CLEAR BUTTON ---
col_title, col_clear = st.columns([0.85, 0.15])
with col_title:
    st.markdown("### 🤖 Local Agentic AI Orchestration Platform")
with col_clear:
    if st.button("🗑️ Clear Chat", use_container_width=True):
        requests.post(f"{BACKEND_URL}/chat/clear")
        st.rerun()

# --- SIDEBAR CONFIGURATION ---
with st.sidebar:
    st.header("⚙️ Global Control Panel")
    
    try:
        models_data = requests.get(f"{BACKEND_URL}/models").json()
        model_names = [m["name"] for m in models_data] if models_data else ["llama3"]
    except Exception:
        model_names = ["llama3 (offline)"]

    selected_model = st.selectbox("🧠 Select AI Model", model_names)
    st.divider()

    try:
        config_data = requests.get(f"{BACKEND_URL}/config").json()
    except Exception:
        config_data = {"mcp_servers": {}, "skills": {}}
        st.error("Could not reach backend FastAPI service.")

    # Section A: MCP Servers
    st.subheader("🌐 Model Context Protocol")
    
    with st.expander("📂 Managed MCP Servers", expanded=True):
        if not config_data["mcp_servers"]:
            st.caption("No MCP servers configured.")
        for name, metadata in config_data["mcp_servers"].items():
            st.markdown(f"**{name}**")
            st.caption(f"`{metadata['command']}` {' '.join(metadata['args'])}")
            
            col1, col2 = st.columns(2)
            with col1:
                # FIX: Changed 'if' to 'with' for the popover container
                with st.popover(f"✏️ Edit"):
                    m_cmd = st.text_input("Binary", value=metadata['command'], key=f"cmd_{name}")
                    m_args = st.text_input("Args", value=",".join(metadata['args']), key=f"args_{name}")
                    if st.button("Save Changes", key=f"save_{name}"):
                        args_list = [a.strip() for a in m_args.split(",") if a.strip()]
                        requests.post(f"{BACKEND_URL}/config/mcp", json={"name": name, "command": m_cmd, "args": args_list})
                        st.rerun()
            with col2:
                if st.button(f"🗑️ Delete", key=f"del_{name}"):
                    requests.delete(f"{BACKEND_URL}/config/mcp/{name}")
                    st.rerun()
            st.divider()
    
    with st.expander("➕ Register New MCP Server"):
        mcp_name = st.text_input("Registry Name", placeholder="git-integration")
        mcp_cmd = st.text_input("Execution Binary", placeholder="npx.cmd")
        mcp_args = st.text_input("Arguments (comma separated)", placeholder="-y, @modelcontextprotocol/server-git")
        if st.button("Mount MCP Server"):
            args_list = [a.strip() for a in mcp_args.split(",") if a.strip()]
            res = requests.post(f"{BACKEND_URL}/config/mcp", json={
                "name": mcp_name, "command": mcp_cmd, "args": args_list
            })
            if res.status_code == 200:
                st.success(f"MCP '{mcp_name}' connected!")
                st.rerun()

    st.divider()

    # Section B: Skills
    st.subheader("🛠️ Extensible Agent Skills")
    
    with st.expander("📂 Managed Skills", expanded=True):
        if not config_data["skills"]:
            st.caption("No custom skills configured.")
        for s_name, s_meta in config_data["skills"].items():
            st.markdown(f"**{s_name}**")
            st.caption(f"*{s_meta['description']}*")
            
            col1, col2 = st.columns(2)
            with col1:
                # FIX: Changed 'if' to 'with' for the popover container
                with st.popover(f"✏️ Edit"):
                    sk_desc = st.text_input("Description", value=s_meta['description'], key=f"sdesc_{s_name}")
                    sk_code = st.text_area("Code", value=s_meta['code'], key=f"scode_{s_name}", height=150)
                    if st.button("Save Changes", key=f"ssave_{s_name}"):
                        requests.post(f"{BACKEND_URL}/config/skill", json={"name": s_name, "description": sk_desc, "code": sk_code})
                        st.rerun()
            with col2:
                if st.button(f"🗑️ Delete", key=f"sdel_{s_name}"):
                    requests.delete(f"{BACKEND_URL}/config/skill/{s_name}")
                    st.rerun()
            st.divider()

    with st.expander("➕ Inject New Skill"):
        new_sk_name = st.text_input("Identifier Key", placeholder="file_zipper")
        new_sk_desc = st.text_input("Intent Description")
        new_sk_code = st.text_area("Python Implementation", value="def execute(**kwargs):\n    # Write logic here\n    return 'Done'", height=150)
        if st.button("Deploy Skill Asset"):
            res = requests.post(f"{BACKEND_URL}/config/skill", json={
                "name": new_sk_name, "description": new_sk_desc, "code": new_sk_code
            })
            if res.status_code == 200:
                st.success(f"Skill '{new_sk_name}' deployed!")
                st.rerun()

# --- MAIN CONVERSATION INTERFACE ---
try:
    runtime_state = requests.get(f"{BACKEND_URL}/state").json()
except Exception:
    runtime_state = {"messages": [], "status": "idle", "pending_tool": None}

for msg in runtime_state["messages"]:
    if msg["role"] != "system":
        # Ensure we are dealing with a string
        content = msg["content"]
        if isinstance(content, dict):
            content = content.get("reply", str(content))
        
        if str(content).strip():
            with st.chat_message(msg["role"]):
                st.markdown(str(content))

if runtime_state["status"] == "awaiting_approval" and runtime_state["pending_tool"]:
    tool_req = runtime_state["pending_tool"]
    st.warning("⚠️ **Execution Authorization Requested!**")
    
    with st.container(border=True):
        st.markdown(f"**Type:** `{tool_req['tool_type'].upper()}`")
        st.markdown(f"**Target System:** `{tool_req['target']}`")
        st.markdown(f"**Action/Method:** `{tool_req['method']}`")
        st.markdown(f"**Arguments Payload:**")
        st.json(tool_req["arguments"])
        st.info(f"**Agent Intent Context:** {tool_req.get('thought', 'No logic supplied.')}")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("✅ Approve & Run Action", use_container_width=True):
                requests.post(f"{BACKEND_URL}/chat/approve")
                st.rerun()
        with col2:
            if st.button("❌ Deny Execution", use_container_width=True):
                requests.post(f"{BACKEND_URL}/chat/reject")
                st.rerun()

if user_query := st.chat_input("Chat with your local AI Agent..."):
    if runtime_state["status"] == "awaiting_approval":
        st.error("Please act on the pending execution parameter above before submitting messages.")
    else:
        with st.chat_message("user"):
            st.markdown(user_query)
        
        with st.spinner(f"Agent ({selected_model}) running calculations..."):
            requests.post(f"{BACKEND_URL}/chat", json={"message": user_query, "model_name": selected_model})
            st.rerun()