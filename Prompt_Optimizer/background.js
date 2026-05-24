import { getEncoding } from './lib/tiktoken.js';

const tokenizer = getEncoding();

const PROMPT_PROFILES = {
  "0": "Clean and structure this prompt by stripping out conversational fluff and redundant boilerplate. Keep all core context, technical requirements, and instructions completely intact. Do not add introductory or concluding remarks. Output only the final clean prompt text.",
  "1": "Condense this prompt aggressively. Remove polite phrasing, excess adverbs, and relational framing. Keep raw technical rules, context keys, and required outputs. Do not explain your changes. Output only the final compressed prompt text directly.",
  "2": "Rewrite this prompt completely into a hyper-efficient 'caveman talk' format using minimalist telegraphic sentences. Strip structural filler words while strictly preserving all key logical parameters, critical nouns, and verbs. Output only the final text.",
  "3": "Compress the text down to short phrases and standard shorthand abbreviations. You must keep all core technical instructions, keys, nouns, and rules completely intact so the prompt remains fully functional. Do not include any conversational introduction, notes, wrapper commentary, or explanation. Output only the raw compressed prompt text directly."
};

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message.action === "getModels") {
    fetchModels().then(sendResponse);
    return true; 
  }
  
  if (message.action === "optimizePrompt") {
    optimizePrompt(message.text, message.level, message.model).then(sendResponse);
    return true;
  }
  
  if (message.action === "countTokens") {
    const counts = tokenizer.encode(message.text || "").length;
    sendResponse({ count: counts });
    return true;
  }
});

async function fetchModels() {
  try {
    const response = await fetch('http://127.0.0.1:11434/api/tags');
    if (!response.ok) throw new Error("Ollama connection down");
    const data = await response.json();
    return { success: true, models: data.models || [] };
  } catch (error) {
    return { success: false, error: error.message };
  }
}

async function optimizePrompt(text, level, model) {
  try {
    const systemInstruction = PROMPT_PROFILES[String(level)] || PROMPT_PROFILES["0"];
    
    const startTag = "### TARGET PROMPT TO COMPRESS ###";
    const endTag = "### FINAL COMPRESSED PROMPT OUTPUT ###";

    const response = await fetch('http://127.0.0.1:11434/api/generate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model: model,
        prompt: `Instructions: ${systemInstruction}\n\n${startTag}\n${text}\n${endTag}`,
        stream: false,
        options: { 
          temperature: 0.0, // Ensures deterministic optimization.
          top_k: 1
        }
      })
    });

    if (!response.ok) throw new Error(`Ollama engine returned code ${response.status}`);
    const data = await response.json();
    
    let finalizedText = data.response || "";

    // --- SANITATION GUARDRAIL ---
    // If the model echoed back the delimiter tags, strip them completely out of the response.
    finalizedText = finalizedText.replace(startTag, "");
    finalizedText = finalizedText.replace(endTag, "");
    
    // Strip surrounding quotes or conversational remnants left by small models.
    finalizedText = finalizedText.replace(/^["']|["']$/g, '').trim();

    // Secondary cleanup logic to catch persistent markdown code fences
    if (finalizedText.startsWith("```")) {
      finalizedText = finalizedText.replace(/^```[a-zA-Z]*\n([\s\S]*?)\n```$/g, '$1').trim();
    }

    // Ultimate fallback if sanitation completely emptied the output string.
    if (!finalizedText || finalizedText.length < 3) {
      finalizedText = text;
    }

    const tokensBefore = tokenizer.encode(text).length;
    const tokensAfter = tokenizer.encode(finalizedText).length;
    const saved = tokensBefore - tokensAfter;

    // Safety Gate: If it didn't actually save any tokens, treat as optimized.
    if (saved <= 0) {
      return {
        success: true,
        optimizedText: text, 
        alreadyOptimized: true,
        tokensBefore,
        tokensAfter: tokensBefore,
        saved: 0,
        percentSaved: 0
      };
    }

    const percentSaved = Math.round((saved / tokensBefore) * 100);

    return {
      success: true,
      optimizedText: finalizedText,
      alreadyOptimized: false,
      tokensBefore,
      tokensAfter,
      saved,
      percentSaved
    };
  } catch (error) {
    return { success: false, error: error.message };
  }
}