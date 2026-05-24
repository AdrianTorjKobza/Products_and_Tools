document.addEventListener('DOMContentLoaded', async () => {
  const modelSelect = document.getElementById('model-select');
  const levelSlider = document.getElementById('level-slider');
  const sliderLabel = document.getElementById('slider-label');
  const errorCard = document.getElementById('connection-error');
  const activeUi = document.getElementById('active-ui');

  const modes = [
    "Clean & Structural (Level 0)",
    "Balanced Compression (Level 1)",
    "Caveman Telegraphic (Level 2)",
    "Hyper-Dense Acronyms (Level 3)"
  ];

  // Load configured states.
  const config = await chrome.storage.local.get(['selectedModel', 'optimizationLevel']);
  if (config.optimizationLevel !== undefined) {
    levelSlider.value = config.optimizationLevel;
    sliderLabel.innerText = `Compression Mode: ${modes[config.optimizationLevel]}`;
  }

  // Poll local Ollama setup state.
  chrome.runtime.sendMessage({ action: "getModels" }, (response) => {
    if (!response || !response.success || response.models.length === 0) {
      errorCard.style.display = "block";
      modelSelect.innerHTML = '<option value="">Unavailable</option>';
      return;
    }

    errorCard.style.display = "none";
    modelSelect.innerHTML = '';
    
    response.models.forEach(m => {
      const opt = document.createElement('option');
      opt.value = m.name;
      opt.innerText = m.name;
      if (config.selectedModel === m.name) opt.selected = true;
      modelSelect.appendChild(opt);
    });

    if (!config.selectedModel && response.models.length > 0) {
      chrome.storage.local.set({ selectedModel: response.models[0].name });
    }
  });

  modelSelect.addEventListener('change', () => {
    chrome.storage.local.set({ selectedModel: modelSelect.value });
  });

  levelSlider.addEventListener('input', () => {
    sliderLabel.innerText = `Compression Mode: ${modes[levelSlider.value]}`;
    chrome.storage.local.set({ optimizationLevel: parseInt(levelSlider.value) });
  });
});