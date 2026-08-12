# ArrestShield-Live Project Tasks

## Phase 1: Environment Setup & Project Configuration
- [x] **Task 1**: Setup project directory structure, initialize python virtual environment, and construct `requirements.txt` with core libraries (torch, transformers, faster-whisper, gliner, fastapi, uvicorn, websockets, numpy, pydantic).
- [ ] **Task 2**: Create basic configuration file `config.py` to manage model parameters, API keys, WebSocket paths, and local thresholds.

## Phase 2: Live/Streaming Audio ASR Pipeline
- [ ] **Task 3**: Implement the Voice Activity Detection (VAD) component using `Silero VAD` to segment incoming live microphone audio streams.
- [ ] **Task 4**: Integrate the streaming ASR logic using `faster-whisper` and the `shunya-labs/zero-stt-hinglish` model to support code-switched Indian language inputs.
- [ ] **Task 5**: Design a streaming WebSocket handler to accept chunked binary audio from a client, run VAD, transcribe speech incrementally under 800ms latency, and yield text segments.

## Phase 3: Tri-State ML Detection Engine
- [ ] **Task 6**: Prepare the Multitask MuRIL dataset loader class for scam/non-scam conversations with custom annotations (manipulation triggers, stage progression).
- [ ] **Task 7**: Define and implement the multi-task model architecture utilizing a shared `google/muril-base-cased` backbone and three task heads (Scam classification, Trigger multi-label sigmoid, Stage progression categorical classification).
- [ ] **Task 8**: Develop the training and validation script (`train_detector.py`) for the Multitask MuRIL classifier, including early stopping and classification metrics.
- [ ] **Task 9**: Implement the Tri-State Sliding Context Window logic to classify current dialogue segments into `SAFE`, `UNCERTAIN`, or `FRAUD` based on risk score fusion calculations.

## Phase 4: Adaptive LLM Honeypot Engine
- [ ] **Task 10**: Define the LLM client wrapper (supporting Ollama/vLLM for local execution of `Qwen2.5-7B-Instruct`) and draft the system prompt template enforcing victim persona dynamics.
- [ ] **Task 11**: Implement the Honeypot State Machine to track and transition the synthetic victim through psychological states (`Confused` -> `Frightened` -> `Cooperative` -> `Stalling`).
- [ ] **Task 12**: Develop the Synthetic Entity Vault to dynamically intercept and swap out real payment coordinates/credentials with pre-generated synthetic decoy data.

## Phase 5: Zero-Shot Threat Extraction
- [ ] **Task 13**: Set up GLiNER entity extraction model pipeline to parse text transcripts for custom threat parameters (UPI IDs, phone numbers, fake badge IDs).
- [ ] **Task 14**: Implement regex-based post-processors to clean up and validate extracted entities (UPI formats, Indian mobile phone patterns, URL endpoints).

## Phase 6: Integration, Server, & Web Dashboard
- [ ] **Task 15**: Build the main `app.py` FastAPI server integrating WebSocket ASR connection, Tri-State ML evaluation, LLM Honeypot activation, and GLiNER extraction.
- [ ] **Task 16**: Create a sleek, modern, glassmorphic HTML/CSS/JS frontend dashboard to stream microphone audio, display real-time transcript chunks, plot risk score gauges, and render threat logs.

## Phase 7: End-to-End Simulation & Verification
- [ ] **Task 17**: Write mock integration tests simulating complete scam conversation sequences to verify detection latency, honeypot transitions, and threat extraction metrics.
