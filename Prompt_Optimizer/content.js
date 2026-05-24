(function() {
  let activeTargetInput = null;
  let optimizationBar = null;

  const TARGET_SELECTORS = [
    'textarea[id*="prompt"]',
    'div[contenteditable="true"][role="textbox"]',
    '#prompt-textarea',
    '.ql-editor[contenteditable="true"]'
  ];

  function findTextarea() {
    for (const selector of TARGET_SELECTORS) {
      const element = document.querySelector(selector);
      if (element) return element;
    }
    return null;
  }

  function injectOptimizationUI() {
    const target = findTextarea();
    if (!target) return;
    
    if (target.parentElement.querySelector('.prompt-optimizer-bar')) return;

    activeTargetInput = target;
    
    optimizationBar = document.createElement('div');
    optimizationBar.className = 'prompt-optimizer-bar';
    optimizationBar.style = `
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 10px;
      padding: 6px 12px;
      background: rgba(30, 31, 32, 0.85);
      border: 1px solid rgba(68, 71, 70, 0.6);
      border-radius: 8px;
      margin-bottom: 8px;
      font-family: system-ui, -apple-system, sans-serif;
      font-size: 12px;
      color: #e3e3e3;
      z-index: 9999;
    `;

    optimizationBar.innerHTML = `
      <div style="display:flex; align-items:center; gap:8px;">
        <button type="button" id="btn-optimize-action" style="display:inline-flex; align-items:center; justify-content:center; background:#a8cdfc; color:#131314; border:none; height:24px; padding:0 12px; border-radius:4px; font-weight:600; cursor:pointer; line-height:1; text-align:center;">Optimize</button>
        <span id="prompt-optimization-status" style="color:#aaa;">Local engine ready</span>
      </div>
      <div id="prompt-saving-metrics" style="display:none; font-weight:500;"></div>
    `;

    target.parentNode.insertBefore(optimizationBar, target);
    document.getElementById('btn-optimize-action').addEventListener('click', executionPipeline);
  }

  async function executionPipeline(e) {
    e.preventDefault();
    e.stopPropagation();

    const statusText = document.getElementById('prompt-optimization-status');
    const metricView = document.getElementById('prompt-saving-metrics');
    const actionBtn = document.getElementById('btn-optimize-action');

    const inputData = activeTargetInput.value || activeTargetInput.innerText || "";
    if (!inputData.trim()) {
      statusText.innerText = "Error: Input text empty";
      return;
    }

    actionBtn.disabled = true;
    actionBtn.style.background = "#555";
    statusText.innerText = "Analyzing prompt structure...";
    metricView.style.display = "none";

    const preferences = await chrome.storage.local.get(['selectedModel', 'optimizationLevel']);
    const targetModel = preferences.selectedModel;
    const compressionRate = preferences.optimizationLevel !== undefined ? preferences.optimizationLevel : 1;

    if (!targetModel) {
      statusText.innerText = "Error: Select the local AI Model in extension menu";
      actionBtn.disabled = false;
      actionBtn.style.background = "#a8cdfc";
      return;
    }

    chrome.runtime.sendMessage({
      action: "optimizePrompt",
      text: inputData,
      level: compressionRate,
      model: targetModel
    }, (result) => {
      actionBtn.disabled = false;
      actionBtn.style.background = "#a8cdfc";

      if (!result || !result.success) {
        statusText.innerText = `Error: ${result?.error || 'Local network lost'}`;
        return;
      }

      if (result.alreadyOptimized) {
        statusText.innerText = "Prompt is already optimized!";
        metricView.innerText = "No changes made (0% tokens saved).";
        metricView.style.color = "#aaa";
        metricView.style.display = "block";
        return;
      }

      if (activeTargetInput.value !== undefined) {
        activeTargetInput.value = result.optimizedText;
      } else {
        activeTargetInput.innerText = result.optimizedText;
      }

      activeTargetInput.dispatchEvent(new Event('input', { bubbles: true }));

      statusText.innerText = "Prompt optimized.";
      metricView.innerText = `Saved ${result.percentSaved}% tokens (${result.tokensBefore} → ${result.tokensAfter})`;
      metricView.style.color = "#8cc78c";
      metricView.style.display = "block";
    });
  }

  const domObserver = new MutationObserver(() => {
    injectOptimizationUI();
  });

  domObserver.observe(document.body, { childList: true, subtree: true });
  injectOptimizationUI();
})();