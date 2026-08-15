// ArrestShield-Live Web Application Client
let ws = null;
let audioContext = null;
let mediaStream = null;
let scriptNode = null;
let isStreaming = false;

// DOM Elements
const btnToggleMic = document.getElementById('btnToggleMic');
const micIcon = document.getElementById('micIcon');
const micBtnLabel = document.getElementById('micBtnLabel');
const btnResetSession = document.getElementById('btnResetSession');
const systemStatusBadge = document.getElementById('systemStatusBadge');
const statusText = document.getElementById('statusText');
const transcriptContainer = document.getElementById('transcriptContainer');
const connectionLatency = document.getElementById('connectionLatency');

const riskScoreValue = document.getElementById('riskScoreValue');
const gaugeProgressCircle = document.getElementById('gaugeProgressCircle');
const triStatePill = document.getElementById('triStatePill');
const predictedStageBadge = document.getElementById('predictedStageBadge');

const trigAuthorityVal = document.getElementById('trigAuthorityVal');
const trigAuthorityBar = document.getElementById('trigAuthorityBar');
const trigUrgencyVal = document.getElementById('trigUrgencyVal');
const trigUrgencyBar = document.getElementById('trigUrgencyBar');
const trigIsolationVal = document.getElementById('trigIsolationVal');
const trigIsolationBar = document.getElementById('trigIsolationBar');
const trigPaymentVal = document.getElementById('trigPaymentVal');
const trigPaymentBar = document.getElementById('trigPaymentBar');

const honeypotStatePill = document.getElementById('honeypotStatePill');
const honeypotUtilityScore = document.getElementById('honeypotUtilityScore');
const decoyDetails = document.getElementById('decoyDetails');
const threatGrid = document.getElementById('threatGrid');
const threatCountBadge = document.getElementById('threatCountBadge');

const canvas = document.getElementById('waveformCanvas');
const canvasCtx = canvas.getContext('2d');

// Initialize Dashboard
document.addEventListener('DOMContentLoaded', () => {
    fetchDecoyCredentials();
    initWebSocket();
    btnToggleMic.addEventListener('click', toggleMicrophoneStream);
    btnResetSession.addEventListener('click', resetSession);
    initWaveformCanvas();
});

// Fetch active decoy credentials
async function fetchDecoyCredentials() {
    try {
        const response = await fetch('/api/decoy_credentials');
        if (response.ok) {
            const data = await response.json();
            decoyDetails.innerHTML = `
                <div><strong>Name:</strong> ${data.name || 'Ramesh Chandra Gupta'}</div>
                <div><strong>Bank:</strong> ${data.bank_name || 'HDFC Bank'} (A/C: ${data.account_number || '50100298471203'})</div>
                <div><strong>UPI ID:</strong> <code style="color: var(--accent-cyan);">${data.upi_id || 'ramesh.gupta52@okaxis'}</code></div>
                <div><strong>Decoy OTP:</strong> <code style="color: var(--color-uncertain);">${data.decoy_otp || '849201'}</code></div>
            `;
        }
    } catch (e) {
        console.warn('Could not fetch decoy credentials:', e);
    }
}

// WebSocket Connection Management
function initWebSocket() {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws/asr`;

    ws = new WebSocket(wsUrl);

    ws.onopen = () => {
        statusText.innerText = 'Connected';
        systemStatusBadge.style.borderColor = 'var(--color-safe)';
    };

    ws.onmessage = (event) => {
        try {
            const data = JSON.parse(event.data);
            if (data.type === 'analysis_turn') {
                renderAnalysisTurn(data);
            }
        } catch (e) {
            console.error('Error parsing WebSocket message:', e);
        }
    };

    ws.onclose = () => {
        statusText.innerText = 'Disconnected';
        systemStatusBadge.style.borderColor = 'var(--color-fraud)';
        setTimeout(initWebSocket, 3000);
    };

    ws.onerror = (err) => {
        console.error('WebSocket Error:', err);
    };
}

// Render Pipeline Turn Data into UI
function renderAnalysisTurn(data) {
    const startTime = Date.now();

    // 1. Add Scammer Transcript Bubble
    if (data.transcript) {
        appendChatBubble('SCAMMER / CALLER', data.transcript, 'chat-scammer');
    }

    // 2. Render Risk Score Gauge & State
    if (data.detection) {
        const riskScore = Math.round((data.detection.risk_score || 0) * 100);
        riskScoreValue.innerText = `${riskScore}%`;

        // Update SVG Gauge Circle (Circumference ~ 440)
        const offset = 440 - (440 * (riskScore / 100));
        gaugeProgressCircle.style.strokeDashoffset = offset;

        const state = (data.detection.state || 'SAFE').toUpperCase();
        triStatePill.innerText = state;
        triStatePill.className = `state-pill state-${state.toLowerCase()}`;

        if (state === 'SAFE') {
            gaugeProgressCircle.style.stroke = 'var(--color-safe)';
        } else if (state === 'UNCERTAIN') {
            gaugeProgressCircle.style.stroke = 'var(--color-uncertain)';
        } else {
            gaugeProgressCircle.style.stroke = 'var(--color-fraud)';
        }

        predictedStageBadge.innerText = `Stage: ${data.detection.predicted_stage || 'monitoring'}`;

        // Triggers
        const trig = data.detection.triggers || {};
        updateTriggerMeter(trigAuthorityVal, trigAuthorityBar, trig.authority || 0);
        updateTriggerMeter(trigUrgencyVal, trigUrgencyBar, trig.urgency || 0);
        updateTriggerMeter(trigIsolationVal, trigIsolationBar, trig.isolation || 0);
        updateTriggerMeter(trigPaymentVal, trigPaymentBar, trig.payment_pressure || 0);
    }

    // 3. Render Honeypot Victim Response
    if (data.honeypot && data.honeypot.active) {
        const state = (data.honeypot.state || 'active').toUpperCase();
        honeypotStatePill.innerText = state;
        honeypotStatePill.className = `state-pill state-fraud`;

        if (data.honeypot.utility_score !== undefined) {
            honeypotUtilityScore.innerText = `Utility Score: ${data.honeypot.utility_score.toFixed(1)}`;
        }

        if (data.honeypot.victim_response) {
            appendChatBubble('VICTIM PERSONA (HONEYPOT)', data.honeypot.victim_response, 'chat-victim');
        }
    }

    // 4. Render Extracted Threats
    const threats = data.cumulative_threats || data.threat_extraction;
    if (threats) {
        renderThreatCards(threats);
    }

    connectionLatency.innerText = `Latency: ${Date.now() - startTime} ms`;
}

function updateTriggerMeter(valElem, barElem, val) {
    const pct = Math.round(val * 100);
    valElem.innerText = `${pct}%`;
    barElem.style.width = `${pct}%`;
}

function appendChatBubble(speaker, text, cssClass) {
    const div = document.createElement('div');
    div.className = `chat-bubble ${cssClass}`;

    const timestamp = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
    div.innerHTML = `
        <div class="speaker-label">
            <span>${speaker}</span>
            <span>${timestamp}</span>
        </div>
        <div>${text}</div>
    `;

    transcriptContainer.appendChild(div);
    transcriptContainer.scrollTop = transcriptContainer.scrollHeight;
}

function renderThreatCards(threats) {
    let cardsHtml = '';
    let totalCount = threats.total_valid_threat_indicators || 0;

    const items = [
        { label: 'UPI Handle', list: threats.upi_ids },
        { label: 'Phone Number', list: threats.phone_numbers },
        { label: 'Police Badge', list: threats.police_badge_ids },
        { label: 'Case Reference', list: threats.case_ids },
        { label: 'Claimed Agency', list: threats.claimed_agencies },
        { label: 'Phishing URL', list: threats.urls }
    ];

    items.forEach(cat => {
        if (cat.list && cat.list.length > 0) {
            cat.list.forEach(val => {
                cardsHtml += `
                    <div class="threat-card">
                        <span class="threat-type">${cat.label}</span>
                        <span class="threat-value">${val}</span>
                    </div>
                `;
            });
        }
    });

    if (!cardsHtml) {
        cardsHtml = `
            <div class="threat-card">
                <span class="threat-type">Status</span>
                <span class="threat-value" style="color: var(--text-dim);">No threats detected yet</span>
            </div>
        `;
    }

    threatGrid.innerHTML = cardsHtml;
    threatCountBadge.innerText = `${totalCount} Indicators`;
}

// Microphone Streaming via Web Audio API
async function toggleMicrophoneStream() {
    if (isStreaming) {
        stopMicrophoneStream();
    } else {
        await startMicrophoneStream();
    }
}

async function startMicrophoneStream() {
    try {
        mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true, video: false });
        audioContext = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });

        const source = audioContext.createMediaStreamSource(mediaStream);
        scriptNode = audioContext.createScriptProcessor(4096, 1, 1);

        scriptNode.onaudioprocess = (e) => {
            if (!isStreaming || ws.readyState !== WebSocket.OPEN) return;
            const inputData = e.inputBuffer.getChannelData(0);
            
            // Convert Float32Array [-1.0, 1.0] to Int16 PCM bytes
            const pcmBuffer = new Int16Array(inputData.length);
            for (let i = 0; i < inputData.length; i++) {
                const s = Math.max(-1, Math.min(1, inputData[i]));
                pcmBuffer[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
            }

            ws.send(pcmBuffer.buffer);
            drawWaveform(inputData);
        };

        source.connect(scriptNode);
        scriptNode.connect(audioContext.destination);

        isStreaming = true;
        micBtnLabel.innerText = 'Stop Mic Stream';
        btnToggleMic.className = 'btn btn-danger';
        micIcon.innerText = '⏹️';
        statusText.innerText = 'Streaming Audio';
    } catch (e) {
        alert('Could not access microphone: ' + e.message);
    }
}

function stopMicrophoneStream() {
    isStreaming = false;
    if (scriptNode) scriptNode.disconnect();
    if (mediaStream) mediaStream.getTracks().forEach(t => t.stop());
    if (audioContext) audioContext.close();

    micBtnLabel.innerText = 'Start Mic Stream';
    btnToggleMic.className = 'btn btn-primary';
    micIcon.innerText = '🎙️';
    statusText.innerText = 'Connected';
}

// Reset Session
async function resetSession() {
    if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ action: 'reset' }));
    }

    transcriptContainer.innerHTML = `
        <div class="chat-bubble chat-scammer" style="opacity: 0.7;">
            <div class="speaker-label"><span>SYSTEM</span><span>RESET COMPLETE</span></div>
            Session reset. Ready for new conversation turns.
        </div>
    `;

    riskScoreValue.innerText = '0%';
    gaugeProgressCircle.style.strokeDashoffset = 440;
    triStatePill.innerText = 'SAFE';
    triStatePill.className = 'state-pill state-safe';
    predictedStageBadge.innerText = 'Stage: Monitoring';

    honeypotStatePill.innerText = 'IDLE';
    honeypotStatePill.className = 'state-pill state-safe';
    honeypotUtilityScore.innerText = 'Utility Score: 0.0';

    threatGrid.innerHTML = `
        <div class="threat-card">
            <span class="threat-type">Status</span>
            <span class="threat-value" style="color: var(--text-dim);">No threats detected yet</span>
        </div>
    `;
    threatCountBadge.innerText = '0 Indicators';
    fetchDecoyCredentials();
}

// Waveform Canvas Visualization
function initWaveformCanvas() {
    canvas.width = canvas.parentElement.clientWidth;
    canvas.height = canvas.parentElement.clientHeight;
    canvasCtx.fillStyle = 'rgba(0, 0, 0, 0.3)';
    canvasCtx.fillRect(0, 0, canvas.width, canvas.height);
}

function drawWaveform(samples) {
    canvasCtx.fillStyle = 'rgba(7, 10, 18, 0.3)';
    canvasCtx.fillRect(0, 0, canvas.width, canvas.height);

    canvasCtx.lineWidth = 2;
    canvasCtx.strokeStyle = '#6366f1';
    canvasCtx.beginPath();

    const sliceWidth = canvas.width / samples.length;
    let x = 0;

    for (let i = 0; i < samples.length; i += 16) {
        const v = samples[i];
        const y = (v + 1) / 2 * canvas.height;

        if (i === 0) canvasCtx.moveTo(x, y);
        else canvasCtx.lineTo(x, y);

        x += sliceWidth * 16;
    }

    canvasCtx.lineTo(canvas.width, canvas.height / 2);
    canvasCtx.stroke();
}

// Preset Scenario Simulation Handler
async function simulatePreset(presetType) {
    let scenarioText = "";
    if (presetType === 'arrest') {
        scenarioText = "Main Mumbai Police Cyber Cell se Officer Sharma speak kar raha hu (Badge #MH-4912). Case #CR-2024-8842 register hua hai. Urgent Payment UPI `rbi.verify@okicici` par transfer karo or 9876543210 call karo.";
    } else if (presetType === 'upi') {
        scenarioText = "Aapka TRAI mobile number disconnect ho jayega. Immediate penalty clearance pay karo UPI `cbi.clearance@paytm` par.";
    } else {
        scenarioText = "Namaste beta! Main Sharma uncle bol raha hu. Gher par sab kaise hain?";
    }

    try {
        const response = await fetch('/api/analyze_text', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text: scenarioText, reset_session: false })
        });

        if (response.ok) {
            const data = await response.json();
            renderAnalysisTurn(data);
        }
    } catch (e) {
        console.error('Simulation error:', e);
    }
}
