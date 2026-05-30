# Local Agentic AI Orchestration Platform

A lightweight, secure, and extensible local Agentic AI Orchestration Platform. Inspired by tools like Claude Code, this platform allows you to build, manage, and interact with autonomous AI Agents entirely on your local machine using Ollama, custom Python Skills, and the Model Context Protocol (MCP).

## Key Features

- **Model Context Protocol (MCP) Integration:** Dynamically spawn and manage local MCP servers (Node.js/Python) to grant your agent access to file systems, Git repositories, databases, and more.
- **Human-in-the-Loop Security:** A strict "Review & Approve" authorization gate pauses execution when the agent attempts to run tools or external commands, ensuring you remain in control of your local environment.
- **Extensible Python Skills:** Inject custom, sandboxed Python code snippets directly from the UI to give your agent specialized computational or data-processing abilities.
- **100% Local AI:** Powered by Ollama's native JSON Mode to ensure highly structured, deterministic tool-calling without relying on cloud APIs.
- **Interactive Web GUI:** A clean, Streamlit-powered dashboard featuring a persistent left-hand control panel for tool management and a fluid chat interface.

## Architecture & Tech Stack

The system is decoupled into three primary layers communicating asynchronously:
1. **Frontend View:** Streamlit Web GUI for rapid UI generation and session state management.
2. **Backend Router:** FastAPI for high-performance, asynchronous execution loops and process management.
3. **Local Tools:** Ollama for local inference, alongside local MCP servers and a Human-in-the-Loop gate.

## Setup & Installation

Follow these steps to get your local environment running.

**Step 1: Install Prerequisites**
Ensure you have the following installed on your machine:
- Python 3.9+
- Node.js (npm/npx)
- Ollama (installed and running in the background)

**Step 2: Pull the Local AI Model**
Open your terminal and pull the default model into your Ollama instance:

    ollama pull llama3

**Step 3: Clone the Repository**
Download the project to your local machine and navigate into the project directory.

**Step 4: Create a Virtual Environment**
It is highly recommended to isolate your Python dependencies. Create and activate a virtual environment:

    python -m venv venv
    source venv/bin/activate

*(Note: If you are using Windows, activate it by running `venv\Scripts\activate`)*

**Step 5: Install Dependencies**
Install the required Python packages using pip:

    pip install -r requirements.txt

## Execution Instructions

The platform requires both the Backend (FastAPI) and Frontend (Streamlit) to run simultaneously. Open two separate terminal windows.

**Terminal 1: Boot the Backend Router**<br>
Start the FastAPI server. This service manages tool execution, Ollama routing, and background MCP processes.

    uvicorn backend.main:app --port 8000 --reload

**Terminal 2: Boot the Web GUI**<br>
Open your second terminal window, ensure your virtual environment is still activated, and launch the Streamlit dashboard.
The dashboard will automatically open in your default browser at `http://localhost:8501`.

    streamlit run frontend/app.py
