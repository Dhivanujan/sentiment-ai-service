// State Management
const appState = {
    activeTab: 'single-tab',
    single: {
        debounceTimer: null,
        emotionChart: null
    },
    batch: {
        file: null,
        results: [],
        filteredResults: [],
        currentPage: 1,
        pageSize: 10,
        doughnutChart: null
    },
    api: {
        lang: 'curl'
    }
};

// DOM Elements
const DOM = {
    // Nav
    navItems: document.querySelectorAll('.nav-item'),
    tabPanels: document.querySelectorAll('.tab-panel'),
    pageTitle: document.getElementById('page-title'),
    pageSubtitle: document.getElementById('page-subtitle'),
    
    // Single Analyze
    textInput: document.getElementById('text-input'),
    charCounter: document.getElementById('char-counter'),
    clearBtn: document.getElementById('clear-btn'),
    modelSelect: document.getElementById('model-select'),
    analyzeBtn: document.getElementById('analyze-btn'),
    resultsCard: document.getElementById('results-card'),
    resultsEmpty: document.getElementById('results-empty-state'),
    resultsData: document.getElementById('results-data-view'),
    sentimentBadge: document.getElementById('sentiment-badge'),
    sentimentConfidence: document.getElementById('sentiment-confidence'),
    inferenceTime: document.getElementById('inference-time'),
    avatarPositive: document.getElementById('avatar-positive-svg'),
    avatarNegative: document.getElementById('avatar-negative-svg'),
    avatarNeutral: document.getElementById('avatar-neutral-svg'),
    subtabBtns: document.querySelectorAll('.subtab-btn'),
    subtabPanels: document.querySelectorAll('.subtab-panel'),
    aspectsList: document.getElementById('aspects-list'),
    sentenceHighlightBox: document.getElementById('sentence-highlight-box'),
    
    // Batch
    dropZone: document.getElementById('drop-zone'),
    fileInput: document.getElementById('batch-file-input'),
    fileInfo: document.getElementById('file-info-container'),
    fileName: document.getElementById('selected-file-name'),
    fileSize: document.getElementById('selected-file-size'),
    removeFileBtn: document.getElementById('remove-file-btn'),
    batchModelSelect: document.getElementById('batch-model-select'),
    processBatchBtn: document.getElementById('process-batch-btn'),
    batchEmpty: document.getElementById('batch-empty-state'),
    batchDashboard: document.getElementById('batch-dashboard'),
    batchTotal: document.getElementById('batch-total'),
    batchPos: document.getElementById('batch-pos'),
    batchNeg: document.getElementById('batch-neg'),
    batchNeu: document.getElementById('batch-neu'),
    searchTable: document.getElementById('search-table-input'),
    tableBody: document.querySelector('#batch-results-table tbody'),
    tablePageInfo: document.getElementById('table-page-info'),
    prevPageBtn: document.getElementById('prev-page-btn'),
    nextPageBtn: document.getElementById('next-page-btn'),
    exportCsvBtn: document.getElementById('export-csv-btn'),
    
    // API Playground
    apiTextInput: document.getElementById('api-text-input'),
    apiModelSelect: document.getElementById('api-model-select'),
    testApiBtn: document.getElementById('test-api-btn'),
    codeTabBtns: document.querySelectorAll('.code-tab-btn'),
    codeSnippetTitle: document.getElementById('code-snippet-title'),
    codeSnippetDisplay: document.getElementById('code-snippet-display'),
    copyCodeBtn: document.getElementById('copy-code-btn'),
    jsonResponseDisplay: document.getElementById('json-response-display'),
    apiStatusBadge: document.getElementById('api-status-badge')
};

// -------------------------------------------------------------
// CORE & TAB NAVIGATION
// -------------------------------------------------------------
DOM.navItems.forEach(item => {
    item.addEventListener('click', () => {
        // Remove active class
        DOM.navItems.forEach(nav => nav.classList.remove('active'));
        DOM.tabPanels.forEach(panel => panel.classList.remove('active'));
        
        // Add active class
        item.classList.add('active');
        const tabId = item.getAttribute('data-tab');
        document.getElementById(tabId).classList.add('active');
        appState.activeTab = tabId;
        
        // Update header headers
        updateHeaderInfo(tabId);
    });
});

function updateHeaderInfo(tabId) {
    const titles = {
        'single-tab': {
            title: 'Single Review Analyzer',
            subtitle: 'Analyze the sentiment, emotions, and specific aspects of individual reviews in real time.'
        },
        'batch-tab': {
            title: 'Batch Dataset Processor',
            subtitle: 'Upload a CSV dataset or plain text reviews to check sentiment distributions and generate classification tables.'
        },
        'playground-tab': {
            title: 'API Integration Playground',
            subtitle: 'Integrate the real-time SentimentAI prediction microservice directly into your own applications.'
        },
        'compare-tab': {
            title: 'Model Comparison & Benchmarks',
            subtitle: 'Compare speed, accuracy, and detailed capabilities across our ML and lexicon-based baseline systems.'
        }
    };
    
    DOM.pageTitle.innerText = titles[tabId].title;
    DOM.pageSubtitle.innerText = titles[tabId].subtitle;
}

// -------------------------------------------------------------
// SINGLE TEXT ANALYZER WORKSPACE
// -------------------------------------------------------------
DOM.textInput.addEventListener('input', () => {
    const text = DOM.textInput.value;
    DOM.charCounter.innerText = `${text.length} chars`;
    
    // Auto-analysis with 700ms debounce
    clearTimeout(appState.single.debounceTimer);
    if (text.trim().length > 3) {
        appState.single.debounceTimer = setTimeout(() => {
            runSingleAnalysis();
        }, 700);
    }
});

DOM.analyzeBtn.addEventListener('click', runSingleAnalysis);

DOM.clearBtn.addEventListener('click', () => {
    DOM.textInput.value = '';
    DOM.charCounter.innerText = '0 chars';
    DOM.resultsEmpty.classList.remove('hidden');
    DOM.resultsData.classList.add('hidden');
});

// Subtabs within Single Analysis Results
DOM.subtabBtns.forEach(btn => {
    btn.addEventListener('click', () => {
        DOM.subtabBtns.forEach(b => b.classList.remove('active'));
        DOM.subtabPanels.forEach(p => p.classList.remove('active'));
        
        btn.classList.add('active');
        const panelId = btn.getAttribute('data-sub');
        document.getElementById(panelId).classList.add('active');
    });
});

async function runSingleAnalysis() {
    const text = DOM.textInput.value.trim();
    if (!text) return;
    
    DOM.analyzeBtn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Analyzing...';
    DOM.analyzeBtn.disabled = true;
    
    const model = DOM.modelSelect.value;
    
    try {
        const url = `/predict?text=${encodeURIComponent(text)}&model_type=${model}`;
        const response = await fetch(url, { method: 'POST' });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const data = await response.json();
        renderSingleResults(data);
        
    } catch (error) {
        console.error("Single Analysis Error: ", error);
        alert("Could not contact the prediction API server. Make sure your FastAPI backend is running.");
    } finally {
        DOM.analyzeBtn.innerHTML = '<i class="fa-solid fa-bolt"></i> Run Analysis';
        DOM.analyzeBtn.disabled = false;
    }
}

function renderSingleResults(data) {
    // Show results panels
    DOM.resultsEmpty.classList.add('hidden');
    DOM.resultsData.classList.remove('hidden');
    
    // Set Badge and Confidence
    const sentiment = data.sentiment_label;
    DOM.sentimentBadge.className = 'badge';
    DOM.sentimentBadge.classList.add(`badge-${sentiment}`);
    DOM.sentimentBadge.innerText = sentiment.toUpperCase();
    
    DOM.sentimentConfidence.innerText = `${(data.confidence * 100).toFixed(1)}%`;
    DOM.inferenceTime.innerText = `${data.latency_seconds.toFixed(4)}s`;
    
    // Avatar Animation Swap
    DOM.avatarPositive.classList.add('hidden');
    DOM.avatarNegative.classList.add('hidden');
    DOM.avatarNeutral.classList.add('hidden');
    
    if (sentiment === 'positive') {
        DOM.avatarPositive.classList.remove('hidden');
    } else if (sentiment === 'negative') {
        DOM.avatarNegative.classList.remove('hidden');
    } else {
        DOM.avatarNeutral.classList.remove('hidden');
    }
    
    // 1. Render Emotion Chart
    renderEmotionChart(data.emotions);
    
    // 2. Render Aspects
    renderAspectsList(data.aspects);
    
    // 3. Render Sentence Highlights
    renderSentenceHighlights(data.sentence_breakdown);
}

function renderEmotionChart(emotions) {
    const labels = Object.keys(emotions).map(e => e.toUpperCase());
    const values = Object.values(emotions);
    
    const colors = {
        'JOY': '#00ffff',     // cyan
        'LOVE': '#8a2be2',    // purple
        'ANGER': '#ff007f',   // magenta
        'SADNESS': '#64748b', // muted gray-blue
        'FEAR': '#f59e0b',    // yellow/orange
        'SURPRISE': '#10b981' // green
    };
    
    const backgroundColors = labels.map(l => colors[l] || '#cbd5e1');
    
    if (appState.single.emotionChart) {
        appState.single.emotionChart.destroy();
    }
    
    const ctx = document.getElementById('emotionChart').getContext('2d');
    appState.single.emotionChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                data: values,
                backgroundColor: backgroundColors,
                borderWidth: 0,
                borderRadius: 4
            }]
        },
        options: {
            indexAxis: 'y',
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            return `Intensity: ${context.parsed.x}%`;
                        }
                    }
                }
            },
            scales: {
                x: {
                    grid: { color: 'rgba(255,255,255,0.03)' },
                    ticks: { color: '#64748b', font: { family: 'Outfit' } },
                    max: 100
                },
                y: {
                    grid: { display: false },
                    ticks: { color: '#cbd5e1', font: { family: 'Outfit', weight: 'bold' } }
                }
            }
        }
    });
}

function renderAspectsList(aspects) {
    DOM.aspectsList.innerHTML = '';
    
    if (!aspects || aspects.length === 0) {
        DOM.aspectsList.innerHTML = '<div class="text-muted small">No distinct aspects (price, quality, service, etc.) identified in this text.</div>';
        return;
    }
    
    aspects.forEach(item => {
        // Map average score [-1, 1] to percent [0, 100]
        const scorePercent = ((item.score + 1) / 2) * 100;
        
        let barColor = 'var(--text-muted)';
        let labelColor = 'var(--text-muted)';
        if (item.sentiment === 'positive') {
            barColor = 'var(--neon-cyan)';
            labelColor = 'var(--neon-cyan)';
        } else if (item.sentiment === 'negative') {
            barColor = 'var(--neon-magenta)';
            labelColor = 'var(--neon-magenta)';
        }
        
        const aspectDiv = document.createElement('div');
        aspectDiv.className = 'aspect-item';
        aspectDiv.innerHTML = `
            <span class="aspect-name">${item.aspect}</span>
            <div class="aspect-meter">
                <div class="aspect-fill" style="width: ${scorePercent}%; background-color: ${barColor};"></div>
            </div>
            <span class="aspect-sentiment-label" style="color: ${labelColor};">${item.sentiment}</span>
        `;
        DOM.aspectsList.appendChild(aspectDiv);
    });
}

function renderSentenceHighlights(sentences) {
    DOM.sentenceHighlightBox.innerHTML = '';
    
    if (!sentences || sentences.length === 0) return;
    
    sentences.forEach(s => {
        const span = document.createElement('span');
        span.className = `highlight-span ${s.sentiment}`;
        span.innerText = s.text + ' ';
        span.title = `Score: ${s.score.toFixed(2)} (${s.sentiment})`;
        DOM.sentenceHighlightBox.appendChild(span);
    });
}

// -------------------------------------------------------------
// BATCH DATASET PROCESSOR
// -------------------------------------------------------------
DOM.dropZone.addEventListener('click', () => DOM.fileInput.click());

DOM.dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    DOM.dropZone.classList.add('dragover');
});

DOM.dropZone.addEventListener('dragleave', () => {
    DOM.dropZone.classList.remove('dragover');
});

DOM.dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    DOM.dropZone.classList.remove('dragover');
    if (e.dataTransfer.files.length > 0) {
        handleSelectedFile(e.dataTransfer.files[0]);
    }
});

DOM.fileInput.addEventListener('change', () => {
    if (DOM.fileInput.files.length > 0) {
        handleSelectedFile(DOM.fileInput.files[0]);
    }
});

function handleSelectedFile(file) {
    const isCsv = file.name.endsWith('.csv');
    const isTxt = file.name.endsWith('.txt');
    
    if (!isCsv && !isTxt) {
        alert("Invalid file format. Please upload a .csv or .txt file.");
        return;
    }
    
    appState.batch.file = file;
    
    // Update uploader layout
    DOM.dropZone.classList.add('hidden');
    DOM.fileInfo.classList.remove('hidden');
    DOM.fileName.innerText = file.name;
    
    // Format size
    const kb = file.size / 1024;
    DOM.fileSize.innerText = kb > 1024 ? `${(kb/1024).toFixed(1)} MB` : `${kb.toFixed(1)} KB`;
    
    DOM.processBatchBtn.disabled = false;
}

DOM.removeFileBtn.addEventListener('click', () => {
    appState.batch.file = null;
    DOM.fileInput.value = '';
    DOM.dropZone.classList.remove('hidden');
    DOM.fileInfo.classList.add('hidden');
    DOM.processBatchBtn.disabled = true;
});

DOM.processBatchBtn.addEventListener('click', async () => {
    if (!appState.batch.file) return;
    
    DOM.processBatchBtn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Processing...';
    DOM.processBatchBtn.disabled = true;
    
    const formData = new FormData();
    formData.append('file', appState.batch.file);
    const model = DOM.batchModelSelect.value;
    
    try {
        const url = `/predict/batch?model_type=${model}`;
        const response = await fetch(url, {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const data = await response.json();
        
        if (data.error) {
            alert(data.error);
            return;
        }
        
        appState.batch.results = data.results;
        appState.batch.filteredResults = [...data.results];
        appState.batch.currentPage = 1;
        
        // Show dashboard panel
        DOM.batchEmpty.classList.add('hidden');
        DOM.batchDashboard.classList.remove('hidden');
        
        // Render Dashboard Metrics
        DOM.batchTotal.innerText = data.total_processed;
        DOM.batchPos.innerText = data.summary.positive;
        DOM.batchNeg.innerText = data.summary.negative;
        DOM.batchNeu.innerText = data.summary.neutral;
        
        // Render doughnut chart
        renderBatchDoughnut(data.summary);
        
        // Render Table
        renderBatchTable();
        
    } catch (e) {
        console.error("Batch processing error: ", e);
        alert("Failed to process dataset. Ensure your CSV has an explicit text column ('review', 'text', 'comment') and headers.");
    } finally {
        DOM.processBatchBtn.innerHTML = '<i class="fa-solid fa-gears"></i> Process Dataset';
        DOM.processBatchBtn.disabled = false;
    }
});

function renderBatchDoughnut(summary) {
    if (appState.batch.doughnutChart) {
        appState.batch.doughnutChart.destroy();
    }
    
    const ctx = document.getElementById('batchDoughnutChart').getContext('2d');
    appState.batch.doughnutChart = new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: ['Positive', 'Negative', 'Neutral'],
            datasets: [{
                data: [summary.positive, summary.negative, summary.neutral],
                backgroundColor: ['#00ffff', '#ff007f', '#64748b'],
                borderColor: '#101725',
                borderWidth: 2
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'bottom',
                    labels: { color: '#cbd5e1', font: { family: 'Outfit' } }
                }
            },
            cutout: '65%'
        }
    });
}

function renderBatchTable() {
    const list = appState.batch.filteredResults;
    const start = (appState.batch.currentPage - 1) * appState.batch.pageSize;
    const end = start + appState.batch.pageSize;
    const pageItems = list.slice(start, end);
    
    DOM.tableBody.innerHTML = '';
    
    if (pageItems.length === 0) {
        DOM.tableBody.innerHTML = `<tr><td colspan="3" class="text-center text-muted">No matching reviews found.</td></tr>`;
        DOM.tablePageInfo.innerText = 'Showing 0-0 of 0';
        DOM.prevPageBtn.disabled = true;
        DOM.nextPageBtn.disabled = true;
        return;
    }
    
    pageItems.forEach(item => {
        const tr = document.createElement('tr');
        
        let badgeColor = 'badge-neutral';
        if (item.sentiment === 'positive') badgeColor = 'badge-positive';
        else if (item.sentiment === 'negative') badgeColor = 'badge-negative';
        
        tr.innerHTML = `
            <td>${escapeHtml(item.text)}</td>
            <td><span class="badge ${badgeColor}">${item.sentiment.toUpperCase()}</span></td>
            <td><code class="text-secondary">${item.vader_score.toFixed(3)}</code></td>
        `;
        DOM.tableBody.appendChild(tr);
    });
    
    // Pagination controls
    DOM.tablePageInfo.innerText = `Showing ${start + 1}-${Math.min(end, list.length)} of ${list.length}`;
    DOM.prevPageBtn.disabled = appState.batch.currentPage === 1;
    DOM.nextPageBtn.disabled = end >= list.length;
}

// Search table
DOM.searchTable.addEventListener('input', () => {
    const q = DOM.searchTable.value.toLowerCase().trim();
    if (q === '') {
        appState.batch.filteredResults = [...appState.batch.results];
    } else {
        appState.batch.filteredResults = appState.batch.results.filter(item => 
            item.text.toLowerCase().includes(q)
        );
    }
    appState.batch.currentPage = 1;
    renderBatchTable();
});

DOM.prevPageBtn.addEventListener('click', () => {
    if (appState.batch.currentPage > 1) {
        appState.batch.currentPage--;
        renderBatchTable();
    }
});

DOM.nextPageBtn.addEventListener('click', () => {
    const totalPages = Math.ceil(appState.batch.filteredResults.length / appState.batch.pageSize);
    if (appState.batch.currentPage < totalPages) {
        appState.batch.currentPage++;
        renderBatchTable();
    }
});

// Client side CSV download export
DOM.exportCsvBtn.addEventListener('click', () => {
    const results = appState.batch.results;
    if (results.length === 0) return;
    
    let csvContent = "data:text/csv;charset=utf-8,";
    csvContent += "ReviewText,PredictedSentiment,VaderScore\n";
    
    results.forEach(r => {
        // Escape quotes
        const cleanedText = r.text.replace(/"/g, '""');
        csvContent += `"${cleanedText}",${r.sentiment},${r.vader_score}\n`;
    });
    
    const encodedUri = encodeURI(csvContent);
    const link = document.createElement("a");
    link.setAttribute("href", encodedUri);
    link.setAttribute("download", `classified_${appState.batch.file.name}`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
});

// Helper: Escape HTML string
function escapeHtml(str) {
    return str
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#039;");
}

// -------------------------------------------------------------
// API PLAYGROUND
// -------------------------------------------------------------
DOM.codeTabBtns.forEach(btn => {
    btn.addEventListener('click', () => {
        DOM.codeTabBtns.forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        const lang = btn.getAttribute('data-lang');
        appState.api.lang = lang;
        updateCodeSnippet();
    });
});

DOM.apiTextInput.addEventListener('input', updateCodeSnippet);
DOM.apiModelSelect.addEventListener('change', updateCodeSnippet);

function updateCodeSnippet() {
    const text = DOM.apiTextInput.value || "Excellent display.";
    const model = DOM.apiModelSelect.value;
    const url = `${window.location.origin}/predict?text=${encodeURIComponent(text)}&model_type=${model}`;
    
    let code = '';
    let title = '';
    
    if (appState.api.lang === 'curl') {
        title = 'cURL Terminal Command';
        code = `curl -X POST "${url}"`;
    } else if (appState.api.lang === 'javascript') {
        title = 'JavaScript Fetch Snippet';
        code = `fetch("${url}", {
  method: "POST"
})
.then(response => response.json())
.then(data => console.log(data))
.catch(error => console.error("Error:", error));`;
    } else if (appState.api.lang === 'python') {
        title = 'Python Requests Snippet';
        code = `import requests

url = "${window.location.origin}/predict"
params = {
    "text": "${text.replace(/"/g, '\\"')}",
    "model_type": "${model}"
}

response = requests.post(url, params=params)
data = response.json()

print(f"Sentiment: {data['sentiment']}")
print(f"Confidence: {data['confidence']}")`;
    }
    
    DOM.codeSnippetTitle.innerText = title;
    DOM.codeSnippetDisplay.innerText = code;
}

DOM.copyCodeBtn.addEventListener('click', () => {
    navigator.clipboard.writeText(DOM.codeSnippetDisplay.innerText);
    const originalText = DOM.copyCodeBtn.innerHTML;
    DOM.copyCodeBtn.innerHTML = '<i class="fa-solid fa-check text-green"></i> Copied!';
    setTimeout(() => {
        DOM.copyCodeBtn.innerHTML = originalText;
    }, 1500);
});

DOM.testApiBtn.addEventListener('click', async () => {
    const text = DOM.apiTextInput.value.trim();
    if (!text) return;
    
    DOM.testApiBtn.disabled = true;
    DOM.testApiBtn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Sending...';
    DOM.apiStatusBadge.className = 'badge badge-sm badge-success';
    DOM.apiStatusBadge.innerText = 'Pending';
    
    const model = DOM.apiModelSelect.value;
    
    try {
        const url = `/predict?text=${encodeURIComponent(text)}&model_type=${model}`;
        const response = await fetch(url, { method: 'POST' });
        
        DOM.apiStatusBadge.innerText = `HTTP ${response.status}`;
        
        if (!response.ok) {
            DOM.apiStatusBadge.className = 'badge badge-sm badge-danger';
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const data = await response.json();
        DOM.jsonResponseDisplay.innerText = JSON.stringify(data, null, 2);
        
    } catch (e) {
        console.error("API Test Error: ", e);
        DOM.jsonResponseDisplay.innerText = JSON.stringify({ error: e.message }, null, 2);
    } finally {
        DOM.testApiBtn.disabled = false;
        DOM.testApiBtn.innerHTML = '<i class="fa-solid fa-paper-plane"></i> Send Request';
    }
});

// Initialize playground setup on load
window.addEventListener('load', () => {
    updateCodeSnippet();
});
