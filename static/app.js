// The browser side: microphone in, speakers out, and a small interface around them.
//
//   microphone -> mic-worklet.js -> WebSocket -> server
//   server -> WebSocket -> player-worklet.js -> speakers
//
// All numbers (sample rates, chunk size) come from the server's "config" message,
// so they live in .env and not in this file.

const toggleButton = document.getElementById("toggle");
const statusPill = document.getElementById("status");
const interimEl = document.getElementById("interim");
const errorEl = document.getElementById("error");
const warningEl = document.getElementById("warning");
const toolsEl = document.getElementById("tools");
const logEl = document.getElementById("log");
const metricsEl = document.getElementById("metrics");
const averageEl = document.getElementById("average");

const STATUS_LABELS = {
    idle: "Idle",
    connecting: "Connecting...",
    listening: "Listening",
    thinking: "Thinking...",
    speaking: "Speaking",
};

let socket = null;
let audioContext = null;
let modulesLoaded = false;
let micStream = null;
let micSource = null;
let micNode = null;
let micMute = null;
let playerNode = null;
let streaming = false; // true once the server says it is ready for microphone audio
let running = false;
let assistantBubble = null;
const perceivedHistory = [];

// ------------------------------------------------------------------ small UI helpers
function setStatus(state) {
    statusPill.textContent = STATUS_LABELS[state] || state;
    statusPill.className = "pill " + state;
}

function showError(message) {
    errorEl.textContent = message;
    errorEl.hidden = false;
}

function clearError() {
    errorEl.hidden = true;
    warningEl.hidden = true;
}

// One line per tool call: the name, whether it worked, and how long it took.
const toolRows = [];
function addToolRow(m) {
    toolRows.push([m.name, m.ok ? "ok" : (m.error || "failed"), m.ms + " ms"]);
    toolsEl.innerHTML = "";
    for (const [name, result, time] of toolRows.slice(-8)) {
        const row = toolsEl.insertRow();
        row.insertCell().textContent = name + " (" + result + ")";
        row.insertCell().textContent = time;
    }
}

function addMessage(role, text) {
    const bubble = document.createElement("div");
    bubble.className = "msg " + role;
    bubble.textContent = text;
    logEl.appendChild(bubble);
    bubble.scrollIntoView({ block: "end" });
    return bubble;
}

function renderMetrics(m) {
    const rows = [
        ["Waited after your last word before the system was sure you were done", m.eot_lag_ms],
        ["Early answer was used (speculative start)", m.draft_hit ? "yes" : "no"],
        ["OpenAI request sent", m.llm_request_ms],
        ["OpenAI first word", m.llm_first_token_ms],
        ["Booking tools started", m.tools_started_ms],
        ["Booking tools finished", m.tools_done_ms],
        ["Short filler phrase spoken while waiting", m.filler_sent_ms],
        ["First phrase sent to the voice", m.first_chunk_sent_ms],
        ["First audio from the voice", m.tts_first_audio_ms],
        ["First audio reached the browser", m.audio_to_browser_ms],
        ["Sound started playing", m.audio_played_ms],
    ];
    metricsEl.innerHTML = "";
    for (const [label, value] of rows) {
        const row = metricsEl.insertRow();
        row.insertCell().textContent = label;
        row.insertCell().textContent = typeof value === "number" ? value + " ms" : (value ?? "n/a");
    }
    if (typeof m.perceived_ms === "number") {
        const row = metricsEl.insertRow();
        row.className = "total";
        row.insertCell().textContent = "Total: you stopped talking until you heard sound";
        row.insertCell().textContent = m.perceived_ms + " ms";
        perceivedHistory.push(m.perceived_ms);
        const last = perceivedHistory.slice(-10);
        const mean = Math.round(last.reduce((a, b) => a + b, 0) / last.length);
        averageEl.textContent = "Average of the last " + last.length + " replies: " + mean + " ms";
    }
}

// ------------------------------------------------------------------ start and stop
async function startConversation() {
    clearError();
    running = true;
    toggleButton.disabled = true;
    setStatus("connecting");
    try {
        if (!audioContext) audioContext = new AudioContext({ latencyHint: "interactive" });
        await audioContext.resume();
        if (!modulesLoaded) {
            await audioContext.audioWorklet.addModule("/static/mic-worklet.js");
            await audioContext.audioWorklet.addModule("/static/player-worklet.js");
            modulesLoaded = true;
        }

        const protocol = location.protocol === "https:" ? "wss" : "ws";
        socket = new WebSocket(protocol + "://" + location.host + "/ws/voice");
        socket.binaryType = "arraybuffer";
        socket.onmessage = onServerMessage;
        socket.onclose = () => { if (running) stopConversation(false); };
        socket.onerror = () => showError("The connection to the server failed.");
    } catch (error) {
        showError("Could not start: " + error.message);
        stopConversation(false);
    }
}

function stopConversation(notifyServer = true) {
    running = false;
    streaming = false;
    if (notifyServer && socket && socket.readyState === WebSocket.OPEN) {
        socket.send(JSON.stringify({ type: "stop" }));
    }
    if (socket) {
        socket.onclose = null;
        socket.close();
        socket = null;
    }
    if (micStream) micStream.getTracks().forEach((track) => track.stop());
    for (const node of [micSource, micNode, micMute, playerNode]) {
        if (node) node.disconnect();
    }
    micStream = micSource = micNode = micMute = playerNode = null;
    interimEl.textContent = "";
    toggleButton.textContent = "Start conversation";
    toggleButton.classList.remove("stop");
    toggleButton.disabled = false;
    setStatus("idle");
}

toggleButton.addEventListener("click", () => {
    if (running) stopConversation(true);
    else startConversation();
});

// ------------------------------------------------------------------ building the audio graph
async function buildAudioGraph(config) {
    // speakers
    playerNode = new AudioWorkletNode(audioContext, "player-processor", {
        numberOfInputs: 0,
        numberOfOutputs: 1,
        outputChannelCount: [1],
        processorOptions: { inputRate: config.tts_sample_rate },
    });
    playerNode.connect(audioContext.destination);
    playerNode.port.onmessage = (event) => {
        const message = event.data;
        if (!socket || socket.readyState !== WebSocket.OPEN) return;
        if (message.type === "started") socket.send(JSON.stringify({ type: "playback_started" }));
        if (message.type === "progress") socket.send(JSON.stringify({ type: "playback_progress", played_ms: Math.round(message.playedMs) }));
        if (message.type === "drained") socket.send(JSON.stringify({ type: "playback_drained" }));
    };

    // microphone (echo cancellation matters: it stops the assistant hearing itself)
    micStream = await navigator.mediaDevices.getUserMedia({
        audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true, autoGainControl: true },
    });
    micSource = audioContext.createMediaStreamSource(micStream);
    micNode = new AudioWorkletNode(audioContext, "mic-processor", {
        numberOfInputs: 1,
        numberOfOutputs: 1,
        outputChannelCount: [1],
        processorOptions: { targetRate: config.mic_sample_rate, chunkMs: config.mic_chunk_ms },
    });
    micMute = audioContext.createGain(); // the mic node must be connected to the output to run; mute it
    micMute.gain.value = 0;
    micSource.connect(micNode);
    micNode.connect(micMute);
    micMute.connect(audioContext.destination);
    micNode.port.onmessage = (event) => {
        if (streaming && socket && socket.readyState === WebSocket.OPEN) socket.send(event.data);
    };
}

// ------------------------------------------------------------------ messages from the server
async function onServerMessage(event) {
    // binary = reply audio: hand it straight to the player
    if (typeof event.data !== "string") {
        if (playerNode) playerNode.port.postMessage({ type: "audio", pcm: event.data }, [event.data]);
        return;
    }

    const message = JSON.parse(event.data);
    switch (message.type) {
        case "config":
            try {
                await buildAudioGraph(message);
                socket.send(JSON.stringify({ type: "start" }));
            } catch (error) {
                showError("Microphone problem: " + error.message);
                stopConversation(false);
            }
            break;
        case "ready":
            streaming = true;
            toggleButton.textContent = "Stop conversation";
            toggleButton.classList.add("stop");
            toggleButton.disabled = false;
            break;
        case "state":
            setStatus(message.value);
            // the server went back to idle by itself (a failed start, a lost connection): reset the buttons too
            if (message.value === "idle" && running) stopConversation(false);
            break;
        case "user_speaking":
            interimEl.textContent = "...";
            break;
        case "transcript_interim":
            interimEl.textContent = message.text;
            break;
        case "transcript_final":
            interimEl.textContent = "";
            addMessage("user", message.text);
            assistantBubble = addMessage("assistant", "");
            break;
        case "tool":
            addToolRow(message);
            break;
        case "warning":
            warningEl.textContent = message.message;
            warningEl.hidden = false;
            break;
        case "assistant_text":
            if (assistantBubble) assistantBubble.textContent += message.text;
            break;
        case "audio_end":
            if (playerNode) playerNode.port.postMessage({ type: "end" });
            break;
        case "clear":
            if (playerNode) playerNode.port.postMessage({ type: "clear" });
            break;
        case "interrupted":
            if (assistantBubble) assistantBubble.classList.add("interrupted");
            break;
        case "metrics":
            renderMetrics(message);
            break;
        case "idle_timeout":
            showError("Stopped listening because nobody spoke for a while. Press Start to continue.");
            break;
        case "error":
            showError(message.message);
            break;
    }
}

window.addEventListener("beforeunload", () => stopConversation(true));