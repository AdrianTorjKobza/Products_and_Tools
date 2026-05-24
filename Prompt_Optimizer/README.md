# Local Prompt Optimizer Browser Extension

A private browser extension tool, that optimizes prompts directly inside ChatGPT or Gemini, to reduce token usage and save API costs, using local Ollama models.

## Optimization Levels
**The original prompt** <br>
<img src="https://github.com/AdrianTorjKobza/Products_and_Tools/blob/3f2a8015126df948161c5893c341892b56af5cb2/Prompt_Optimizer/screenshots/1.original_prompt.png"></img><br><br>
**Clean & Structural (Level 0)**<br>
<img src="https://github.com/AdrianTorjKobza/Products_and_Tools/blob/3f2a8015126df948161c5893c341892b56af5cb2/Prompt_Optimizer/screenshots/2.level0_optimization.png"></img><br><br>
**Balanced Compression (Level 1)**<br>
<img src="https://github.com/AdrianTorjKobza/Products_and_Tools/blob/3f2a8015126df948161c5893c341892b56af5cb2/Prompt_Optimizer/screenshots/3.level1_optimization.png"></img><br><br>
**Caveman Telegraphic (Level 2)**<br>
<img src="https://github.com/AdrianTorjKobza/Products_and_Tools/blob/3f2a8015126df948161c5893c341892b56af5cb2/Prompt_Optimizer/screenshots/4.level2_optimization.png"></img><br><br>
**Hyper-Dense Acronyms (Level 3)**<br>
<img src="https://github.com/AdrianTorjKobza/Products_and_Tools/blob/3f2a8015126df948161c5893c341892b56af5cb2/Prompt_Optimizer/screenshots/5.level3_optimization.png"></img>

## Technical Highlights
* **Local In-Context Computation:** Uses an isolated runtime via Manifest V3 Service Workers interfacing with local **Ollama** instances. Your prompt metrics never leak to third-party servers.
* **Granular Optimization Scaling:** Features 4 configurable operational tiers from structural cleanup down to extreme telegraphic token serialization models.
* **Native Context Adaptation:** Injects control wrappers transparently into interface elements using mutation listeners. This avoids UI flickering on complex reactive SPAs like React or Next.js.
* **Onboard Token Analysis:** Uses native JavaScript byte-pair token estimation to show before-and-after tracking results in real time.

## Tech Stack

This extension is built entirely on a **zero-dependency, native web stack** to guarantee rapid runtime execution, eliminate heavy build-step tooling, and ensure straightforward open-source code audits.

* **Extension Framework:** WebExtensions Manifest V3 API (Fully compatible with Chromium browsers: Chrome, Edge, Brave, Opera).
* **Core Languages:** Vanilla ECMAScript 2022 (Asynchronous JavaScript), HTML5, and isolated CSS3 variables.
* **Tokenizer Engine:** Customized implementation of the `cl100k_base` Byte-Pair Encoding (BPE) algorithm for offline token estimations matching OpenAI’s sub-word distributions.
* **Local LLM Integration Layer:** Native Fetch API piping directly into the local loopback port of the Ollama orchestration engine.

### System Architecture

To preserve absolute privacy and maintain strict security isolation, the extension utilizes a decoupled, asynchronous, 3-tier browser architecture:

```text
┌────────────────────────────────────────────────────────┐
│                   Chromium Sandbox                     │
│                                                        │
│  ┌────────────────┐           ┌─────────────────────┐  │
│  │  content.js    │           │     popup.html      │  │
│  │ (DOM Injection)│           │ (Model/Mode Config) │  │
│  └───────┬────────┘           └──────────┬──────────┘  │
│          │                               │             │
│          │ Chrome Message Passing        │ Storage API │
│          ▼                               ▼             │
│  ┌──────────────────────────────────────────────────┐  │
│  │                  background.js                   │  │
│  │    (Service Worker / BPE Tokenizer / Cors Hub)   │  │
│  └───────────────────────┬──────────────────────────┘  │
└──────────────────────────┼─────────────────────────────┘
                           │
                           │ Local HTTP Pipeline
                           ▼
               ┌───────────────────────────┐
               │  Local Ollama Engine      │
               │  (http://127.0.0.1:11434) │
               └───────────────────────────┘
```

## Setup & Installation

### Configure the Local Engine Dependency
The runtime depends directly on a functioning local Ollama server deployment.
* Download the executable bundle targeting your host engine system from [Ollama.com](https://ollama.com).
* Spin up your preferred lightweight processing engine model inside your terminal workspace shell:
```bash
   ollama run llama3
```

### Load the Extension in Chrome
* Clone this repository workspace folder structure onto your system.
* Open Chrome browser and navigate to `chrome://extensions/`.
* Locate the Developer mode slider switch (upper-right corner) and set it to Active.
* Click on the `Load unpacked` button, located in the top action bar menu.
* Select the the folder directory containing the project.

## Usage Guide
* Navigate directly into either ChatGPT or Gemini.
* An `Optimize` interaction command utility bar will render above the conversational input panel area.
* Access the browser toolbar tray icon configuration settings interface anytime to choose a specific AI Model or the optimization processing level.
* Compose your prompt inside ChatGPT or Gemini, then trigger the `Optimize` button to compress the prompt and view token savings before sending it.

## Security & Sandboxing Structure
The extension strictly utilizes local network host system bindings exclusively (127.0.0.1). Content Scripts run completely isolated from webpage code executions, rendering them fully immune against host document scripting access issues or external data mining routines.
