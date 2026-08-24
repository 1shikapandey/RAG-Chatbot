document.addEventListener("DOMContentLoaded", () => {
    // State Variables
    let rawDocuments = {};
    let configStatus = { has_api_key: false };
    let metricsChart = null;

    // DOM Elements
    const appStatus = document.getElementById("app-status");
    const toggleConfig = document.getElementById("toggle-config");
    const configFields = document.getElementById("config-fields");
    const configStatusLabel = document.getElementById("config-status-label");
    const apiKeyInput = document.getElementById("api-key-input");
    const saveKeyBtn = document.getElementById("save-key-btn");
    const apiKeyBar = document.getElementById("api-key-bar");

    const chatForm = document.getElementById("chat-form");
    const chatInput = document.getElementById("chat-input");
    const chatMessages = document.getElementById("chat-messages");
    const sendBtn = document.getElementById("send-btn");

    const tabButtons = document.querySelectorAll(".tab-btn");
    const tabContents = document.querySelectorAll(".tab-content");

    const docSelect = document.getElementById("doc-select");
    const openedDocTitle = document.getElementById("opened-doc-title");
    const openedDocLines = document.getElementById("opened-doc-lines");
    const docContentView = document.getElementById("doc-content-view");

    const runEvalBtn = document.getElementById("run-eval-btn");
    const evalLoading = document.getElementById("eval-loading");
    const evalResultsContainer = document.getElementById("eval-results-container");
    const evalTableBody = document.getElementById("eval-table-body");

    // Metrics Values
    const valPrecision = document.getElementById("val-precision");
    const valRecall = document.getElementById("val-recall");
    const valRelevancy = document.getElementById("val-relevancy");
    const valFaithfulness = document.getElementById("val-faithfulness");

    // Modal elements
    const detailsModal = document.getElementById("details-modal");
    const closeModalBtn = document.getElementById("close-modal");
    const modalItemId = document.getElementById("modal-item-id");
    const modalQ = document.getElementById("modal-q");
    const modalRef = document.getElementById("modal-ref");
    const modalGen = document.getElementById("modal-gen");
    const modalExpectedSrc = document.getElementById("modal-expected-src");
    const modalActualSrc = document.getElementById("modal-actual-src");
    const modalMPrecision = document.getElementById("modal-m-precision");
    const modalERecall = document.getElementById("modal-e-recall");
    const modalMRecall = document.getElementById("modal-m-recall");
    const modalERecallText = document.getElementById("modal-e-recall");
    const modalMRelevancy = document.getElementById("modal-m-relevancy");
    const modalERelevancy = document.getElementById("modal-e-relevancy");
    const modalMFaithfulness = document.getElementById("modal-m-faithfulness");
    const modalEFaithfulness = document.getElementById("modal-e-faithfulness");
    const modalEPrecisionText = document.getElementById("modal-e-precision");

    // ----------------------------------------------------
    // 1. INIT & CONFIG CHECK
    // ----------------------------------------------------
    async function checkConfig() {
        try {
            const res = await fetch("/api/config");
            configStatus = await res.json();
            updateConfigUI();
        } catch (e) {
            console.error("Error checking config:", e);
        }
    }

    function updateConfigUI() {
        if (configStatus.has_api_key) {
            appStatus.className = "status-badge connected";
            appStatus.innerHTML = '<i class="fa-solid fa-circle-dot"></i> Connected to Gemini';
            configStatusLabel.className = "config-indicator connected";
            configStatusLabel.textContent = "API Key Active";
            apiKeyBar.classList.add("active");
        } else {
            appStatus.className = "status-badge mock";
            appStatus.innerHTML = '<i class="fa-solid fa-circle-dot"></i> Offline Mock Mode';
            configStatusLabel.className = "config-indicator";
            configStatusLabel.textContent = "Mock Mode";
            apiKeyBar.classList.remove("active");
        }
    }

    // Toggle API Key section
    toggleConfig.addEventListener("click", () => {
        configFields.classList.toggle("show");
    });

    // Save API Key
    saveKeyBtn.addEventListener("click", async () => {
        const key = apiKeyInput.value.trim();
        if (!key) {
            alert("Please enter a valid key.");
            return;
        }

        saveKeyBtn.disabled = true;
        saveKeyBtn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Saving...';

        try {
            const res = await fetch("/api/save-key", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ key })
            });
            const data = await res.json();
            if (data.success) {
                alert(data.message);
                apiKeyInput.value = "";
                configFields.classList.remove("show");
                await checkConfig();
                // Reload documents and build retrievers again
                await fetchDocuments();
            } else {
                alert("Error: " + data.error);
            }
        } catch (e) {
            alert("Error sending key to server.");
        } finally {
            saveKeyBtn.disabled = false;
            saveKeyBtn.innerHTML = '<i class="fa-solid fa-floppy-disk"></i> Save Key';
        }
    });

    // ----------------------------------------------------
    // 2. KNOWLEDGE BASE INSPECTOR
    // ----------------------------------------------------
    async function fetchDocuments() {
        try {
            const res = await fetch("/api/documents");
            rawDocuments = await res.json();
            populateDocDropdown();
        } catch (e) {
            console.error("Error fetching documents:", e);
        }
    }

    function populateDocDropdown() {
        // Clear previous options except placeholder
        docSelect.innerHTML = '<option value="">-- Select a document --</option>';
        Object.keys(rawDocuments).sort().forEach(filename => {
            const opt = document.createElement("option");
            opt.value = filename;
            opt.textContent = filename;
            docSelect.appendChild(opt);
        });
    }

    docSelect.addEventListener("change", (e) => {
        renderDocument(e.target.value);
    });

    function renderDocument(filename, highlightStart = null, highlightEnd = null) {
        if (!filename || !rawDocuments[filename]) {
            openedDocTitle.textContent = "No document selected";
            openedDocLines.textContent = "0 lines";
            docContentView.innerHTML = "Select a document from the dropdown above, or click on a RAG citation source in the chat window to inspect facts and check grounding.";
            return;
        }

        const rawText = rawDocuments[filename];
        const lines = rawText.split(/\r?\n/);
        
        openedDocTitle.textContent = filename;
        openedDocLines.textContent = `${lines.length} lines`;

        let codeHtml = "";
        lines.forEach((lineText, index) => {
            const lineNum = index + 1;
            const isHighlighted = highlightStart !== null && highlightEnd !== null && lineNum >= highlightStart && lineNum <= highlightEnd;
            const highlightClass = isHighlighted ? "highlight" : "";
            
            codeHtml += `<div class="doc-line ${highlightClass}" id="line-${filename}-${lineNum}">
                <span class="doc-line-num">${lineNum}</span>
                <span class="doc-line-text">${escapeHtml(lineText)}</span>
            </div>`;
        });

        docContentView.innerHTML = codeHtml;

        // Scroll highlight into view if active
        if (highlightStart !== null) {
            setTimeout(() => {
                const targetLine = document.getElementById(`line-${filename}-${highlightStart}`);
                if (targetLine) {
                    targetLine.scrollIntoView({ behavior: "smooth", block: "center" });
                }
            }, 100);
        }
    }

    // ----------------------------------------------------
    // 3. TABS SWITCHING
    // ----------------------------------------------------
    tabButtons.forEach(btn => {
        btn.addEventListener("click", () => {
            const tabId = btn.getAttribute("data-tab");
            
            tabButtons.forEach(b => b.classList.remove("active"));
            tabContents.forEach(c => c.classList.remove("active"));
            
            btn.classList.add("active");
            document.getElementById(tabId).classList.add("active");
        });
    });

    // ----------------------------------------------------
    // 4. CHAT SYSTEM
    // ----------------------------------------------------
    chatForm.addEventListener("submit", async (e) => {
        e.preventDefault();
        const query = chatInput.value.trim();
        if (!query) return;

        // Clear input
        chatInput.value = "";

        // Append user message
        appendMessage("user", `<p>${escapeHtml(query)}</p>`);

        // Add loading bubble
        const loadingId = appendLoadingBubble();

        // Disable input
        chatInput.disabled = true;
        sendBtn.disabled = true;

        try {
            const res = await fetch("/api/chat", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ query })
            });
            const data = await res.json();
            
            // Remove loading bubble
            removeBubble(loadingId);

            if (data.error) {
                appendMessage("assistant", `<p class="text-danger">Error: ${escapeHtml(data.error)}</p>`);
            } else {
                let answerHtml = formatMarkdown(data.answer);
                
                // If sources exist, format citation tags
                if (data.sources && data.sources.length > 0) {
                    answerHtml += `<div class="citations-box">
                        <div class="citations-title"><i class="fa-solid fa-folder-open"></i> Grounding Sources</div>
                        <div class="citations-list">`;
                    
                    data.sources.forEach(src => {
                        answerHtml += `<span class="citation-tag" onclick="window.inspectSource('${src.source}', ${src.start_line}, ${src.end_line})">
                            <i class="fa-solid fa-file-text"></i> ${src.source} [L${src.start_line}-${src.end_line}]
                        </span>`;
                    });
                    
                    answerHtml += `</div></div>`;
                }

                appendMessage("assistant", answerHtml);
            }
        } catch (err) {
            removeBubble(loadingId);
            appendMessage("assistant", `<p class="text-danger">Failed to send query. Make sure backend is running.</p>`);
        } finally {
            chatInput.disabled = false;
            sendBtn.disabled = false;
            chatInput.focus();
        }
    });

    function appendMessage(sender, contentHtml) {
        const msgDiv = document.createElement("div");
        msgDiv.className = `message ${sender}-msg`;
        
        const avatar = sender === "user" ? '<i class="fa-solid fa-user"></i>' : '<i class="fa-solid fa-robot"></i>';
        
        msgDiv.innerHTML = `
            <div class="message-avatar">${avatar}</div>
            <div class="message-content">${contentHtml}</div>
        `;
        
        chatMessages.appendChild(msgDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    function appendLoadingBubble() {
        const id = "loading-" + Date.now();
        const msgDiv = document.createElement("div");
        msgDiv.className = "message assistant-msg";
        msgDiv.id = id;
        msgDiv.innerHTML = `
            <div class="message-avatar"><i class="fa-solid fa-robot"></i></div>
            <div class="message-content">
                <div class="spinner" style="width: 20px; height: 20px; border-width: 2px;"></div>
            </div>
        `;
        chatMessages.appendChild(msgDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight;
        return id;
    }

    function removeBubble(id) {
        const bubble = document.getElementById(id);
        if (bubble) bubble.remove();
    }

    // Global utility window functions for onclick references
    window.inspectSource = (filename, startLine, endLine) => {
        // Switch to Document Inspector Tab
        const inspectorTabBtn = document.querySelector('[data-tab="inspector-tab"]');
        inspectorTabBtn.click();
        
        // Select dropdown and render
        docSelect.value = filename;
        renderDocument(filename, startLine, endLine);
    };

    // ----------------------------------------------------
    // 5. EVALUATION SUITE
    // ----------------------------------------------------
    runEvalBtn.addEventListener("click", async () => {
        runEvalBtn.disabled = true;
        evalLoading.classList.remove("hidden");
        evalResultsContainer.classList.add("hidden");

        try {
            const res = await fetch("/api/evaluate", { method: "POST" });
            const data = await res.json();
            
            if (data.error) {
                alert("Evaluation error: " + data.error);
                evalLoading.classList.add("hidden");
                runEvalBtn.disabled = false;
                return;
            }

            renderEvalResults(data);
        } catch (e) {
            alert("Error running evaluation suite.");
            evalLoading.classList.add("hidden");
            runEvalBtn.disabled = false;
        }
    });

    function renderEvalResults(report) {
        evalLoading.classList.add("hidden");
        evalResultsContainer.classList.remove("hidden");
        runEvalBtn.disabled = false;

        const summary = report.summary;

        // Render Values
        valPrecision.textContent = `${Math.round(summary.average_context_precision * 100)}%`;
        valRecall.textContent = `${Math.round(summary.average_context_recall * 100)}%`;
        valRelevancy.textContent = `${Math.round(summary.average_answer_relevancy * 100)}%`;
        valFaithfulness.textContent = `${Math.round(summary.average_faithfulness * 100)}%`;

        // Load Chart.js Visualization
        initChart(
            summary.average_context_precision,
            summary.average_context_recall,
            summary.average_answer_relevancy,
            summary.average_faithfulness
        );

        // Populate detailed table runs
        evalTableBody.innerHTML = "";
        
        window.evalDetailedItems = report.detailed_results; // Save to window context for modal search

        report.detailed_results.forEach(res => {
            const row = document.createElement("tr");
            
            const metrics = res.metrics;
            const categoryClass = res.category.toLowerCase();
            
            // Format inline scores
            const scoreHtml = `
                <div class="mini-metrics">
                    <span class="m-badge ${getScoreClass(metrics.context_precision)}">P: ${metrics.context_precision.toFixed(1)}</span>
                    <span class="m-badge ${getScoreClass(metrics.context_recall)}">R: ${metrics.context_recall.toFixed(1)}</span>
                    <span class="m-badge ${getScoreClass(metrics.answer_relevancy)}">Rel: ${metrics.answer_relevancy.toFixed(1)}</span>
                    <span class="m-badge ${getScoreClass(metrics.faithfulness)}">F: ${metrics.faithfulness.toFixed(1)}</span>
                </div>
            `;

            row.innerHTML = `
                <td><strong>${res.id}</strong></td>
                <td><span class="eval-category ${categoryClass}">${res.category}</span></td>
                <td><span class="text-truncate">${escapeHtml(res.question)}</span></td>
                <td>${scoreHtml}</td>
                <td>
                    <button class="inspect-btn" onclick="window.showEvalDetails('${res.id}')">
                        <i class="fa-solid fa-eye"></i>
                    </button>
                </td>
            `;

            evalTableBody.appendChild(row);
        });
    }

    function getScoreClass(val) {
        if (val >= 0.8) return "pass";
        if (val >= 0.5) return "warn";
        return "fail";
    }

    function initChart(precision, recall, relevancy, faithfulness) {
        if (metricsChart) {
            metricsChart.destroy();
        }

        const ctx = document.getElementById('metricsChart').getContext('2d');
        
        Chart.defaults.color = '#9ca3af';
        Chart.defaults.font.family = "'Plus Jakarta Sans', sans-serif";

        metricsChart = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: ['Context Precision', 'Context Recall', 'Answer Relevancy', 'Faithfulness (Groundedness)'],
                datasets: [{
                    label: 'Score (0.0 - 1.0)',
                    data: [precision, recall, relevancy, faithfulness],
                    backgroundColor: [
                        'rgba(96, 165, 250, 0.4)', // Precision Blue
                        'rgba(192, 132, 252, 0.4)', // Recall Purple
                        'rgba(251, 113, 133, 0.4)', // Relevancy Rose
                        'rgba(45, 212, 191, 0.4)'  // Faithfulness Teal
                    ],
                    borderColor: [
                        '#60a5fa',
                        '#c084fc',
                        '#fb7185',
                        '#2dd4bf'
                    ],
                    borderWidth: 2,
                    borderRadius: 8
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: false
                    }
                },
                scales: {
                    y: {
                        min: 0,
                        max: 1,
                        grid: {
                            color: 'rgba(255, 255, 255, 0.05)'
                        }
                    },
                    x: {
                        grid: {
                            display: false
                        }
                    }
                }
            }
        });
    }

    // Modal Trigger
    window.showEvalDetails = (id) => {
        const item = window.evalDetailedItems.find(i => i.id === id);
        if (!item) return;

        modalItemId.textContent = item.id;
        modalQ.textContent = item.question;
        modalRef.innerHTML = formatMarkdown(item.reference_answer);
        modalGen.innerHTML = formatMarkdown(item.generated_answer);
        
        modalExpectedSrc.textContent = item.expected_sources.length > 0 ? item.expected_sources.join(", ") : "None";
        modalActualSrc.textContent = item.actual_sources.length > 0 ? item.actual_sources.join(", ") : "None";

        // Metrics values
        modalMPrecision.textContent = item.metrics.context_precision.toFixed(2);
        modalMRecall.textContent = item.metrics.context_recall.toFixed(2);
        modalMRelevancy.textContent = item.metrics.answer_relevancy.toFixed(2);
        modalMFaithfulness.textContent = item.metrics.faithfulness.toFixed(2);

        // Explanations texts
        modalEPrecisionText.textContent = item.explanations.context_precision || "No explanation provided.";
        modalERecallText.textContent = item.explanations.context_recall || "No explanation provided.";
        modalERelevancy.textContent = item.explanations.answer_relevancy || "No explanation provided.";
        modalEFaithfulness.textContent = item.explanations.faithfulness || "No explanation provided.";

        detailsModal.classList.add("show");
    };

    closeModalBtn.addEventListener("click", () => {
        detailsModal.classList.remove("show");
    });

    window.addEventListener("click", (e) => {
        if (e.target === detailsModal) {
            detailsModal.classList.remove("show");
        }
    });

    // ----------------------------------------------------
    // HELPER FORMATTING FUNCTIONS
    // ----------------------------------------------------
    function escapeHtml(text) {
        return text
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(/"/g, "&quot;")
            .replace(/'/g, "&#039;");
    }

    function formatMarkdown(text) {
        if (!text) return "";
        let formatted = escapeHtml(text);
        
        // Bold formatting **text**
        formatted = formatted.replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>");
        
        // Inline code formatting `code`
        formatted = formatted.replace(/`(.*?)`/g, "<code class='inline-code'>$1</code>");
        
        // Source bracket highlights e.g. [Source: domains.txt]
        formatted = formatted.replace(/\[Source:\s*(.*?)\]/g, "<span class='inline-source'>[Source: $1]</span>");

        // Convert double newlines to paragraphs
        const paragraphs = formatted.split(/\n\n+/);
        return paragraphs.map(p => `<p>${p.replace(/\n/g, "<br>")}</p>`).join("");
    }

    // Run Startup Checks
    checkConfig();
    fetchDocuments();
});
