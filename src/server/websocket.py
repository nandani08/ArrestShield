import json
import logging
from typing import Optional, Any, Dict
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from src.asr.streaming import StreamingASRProcessor
from src.detection.tristate_detector import TriStateDetector
from src.extraction.gliner_extractor import GLiNERThreatExtractor
from src.extraction.post_processor import ThreatPostProcessor
from src.honeypot.state_machine import HoneypotStateMachine
from src.honeypot.entity_vault import SyntheticEntityVault
from src.config import settings

logger = logging.getLogger("ArrestShield.WebSocket")
router = APIRouter()
ws_router = router


class IntegratedPipelineSession:
    """
    Manages state for an active streaming conversation session:
    ASR -> Tri-State Detection -> GLiNER Threat Extraction -> LLM Honeypot.
    """
    def __init__(
        self,
        asr_processor: Optional[StreamingASRProcessor] = None,
        detector: Optional[TriStateDetector] = None,
        extractor: Optional[GLiNERThreatExtractor] = None,
        post_processor: Optional[ThreatPostProcessor] = None,
        honeypot_machine: Optional[HoneypotStateMachine] = None,
        entity_vault: Optional[SyntheticEntityVault] = None
    ):
        self.asr_processor = asr_processor if asr_processor else StreamingASRProcessor(
            sample_rate=settings.asr.sample_rate,
            window_duration_sec=settings.asr.window_duration_sec,
            latency_target_ms=settings.asr.latency_target_ms
        )
        self.detector = detector if detector else TriStateDetector(model_name="dummy")
        self.extractor = extractor if extractor else GLiNERThreatExtractor(model_name="dummy")
        self.post_processor = post_processor if post_processor else ThreatPostProcessor()
        self.honeypot_machine = honeypot_machine if honeypot_machine else HoneypotStateMachine()
        self.entity_vault = entity_vault if entity_vault else SyntheticEntityVault()

        self.cumulative_extracted_entities: Dict[str, Any] = {
            "upi_ids": [],
            "phone_numbers": [],
            "urls": [],
            "police_badge_ids": [],
            "case_ids": [],
            "claimed_agencies": [],
            "total_valid_threat_indicators": 0
        }

    def reset(self):
        """Resets all pipeline session buffers."""
        self.asr_processor.reset()
        self.detector.reset()
        self.honeypot_machine.reset()
        self.entity_vault.reset_active_decoys()
        self.cumulative_extracted_entities = {
            "upi_ids": [],
            "phone_numbers": [],
            "urls": [],
            "police_badge_ids": [],
            "case_ids": [],
            "claimed_agencies": [],
            "total_valid_threat_indicators": 0
        }

    def process_text_turn(self, transcript_text: str, timestamp_ms: int = 0) -> Dict[str, Any]:
        """
        Executes complete end-to-end analysis on a text turn.
        """
        # 1. Tri-State ML Detection
        detection_result = self.detector.evaluate_turn(transcript_text)

        # 2. Zero-Shot Threat Extraction
        raw_entities = self.extractor.extract_threat_entities(transcript_text)
        threat_report = self.post_processor.process_extracted_threats(raw_entities, transcript_text=transcript_text)

        # Update cumulative threat indicators
        for key in ["upi_ids", "phone_numbers", "urls", "police_badge_ids", "case_ids", "claimed_agencies"]:
            existing_set = set(self.cumulative_extracted_entities[key])
            existing_set.update(threat_report[key])
            self.cumulative_extracted_entities[key] = sorted(list(existing_set))

        self.cumulative_extracted_entities["total_valid_threat_indicators"] = sum(
            len(self.cumulative_extracted_entities[k]) for k in ["upi_ids", "phone_numbers", "urls", "police_badge_ids", "case_ids", "claimed_agencies"]
        )

        # 3. Adaptive LLM Honeypot Activation (Activates on UNCERTAIN or FRAUD)
        is_suspicious = (detection_result["state"] in ["FRAUD", "UNCERTAIN"] or detection_result["risk_score"] >= 0.40)
        honeypot_payload = {"active": False}

        if is_suspicious or self.honeypot_machine.turn_count > 0:
            decoy_context = self.entity_vault.get_decoy_context_string()
            honeypot_turn = self.honeypot_machine.generate_honeypot_turn(
                scammer_input=transcript_text,
                detection_result=detection_result,
                decoy_context=decoy_context,
                new_extracted_count=threat_report["total_valid_threat_indicators"]
            )
            honeypot_payload = {
                "active": True,
                "state": honeypot_turn["state"],
                "victim_response": honeypot_turn["victim_response"],
                "turn_count": honeypot_turn["turn_count"],
                "utility_score": honeypot_turn["utility_score"]
            }

        return {
            "type": "analysis_turn",
            "transcript": transcript_text,
            "timestamp_ms": timestamp_ms,
            "detection": detection_result,
            "threat_extraction": threat_report,
            "cumulative_threats": self.cumulative_extracted_entities,
            "honeypot": honeypot_payload
        }


@router.websocket(settings.server.ws_asr_path)
async def websocket_asr_endpoint(websocket: WebSocket):
    """
    Streaming WebSocket endpoint for real-time speech transcription,
    Tri-State scam risk scoring, threat extraction, and adaptive LLM honeypot engagement.
    """
    await websocket.accept()
    logger.info(f"WebSocket client connected to endpoint '{settings.server.ws_asr_path}'")

    session: Optional[IntegratedPipelineSession] = None
    try:
        session = IntegratedPipelineSession()
    except Exception as e:
            logger.error(f"Failed to initialize IntegratedPipelineSession: {e}")
            await websocket.send_json({"type": "error", "message": f"Initialization error: {str(e)}"})
            await websocket.close(code=1011)
            return

    try:
        while True:
            message = await websocket.receive()

            # 1. Handle Binary Audio Frame (PCM)
            if "bytes" in message and message["bytes"]:
                binary_chunk = message["bytes"]
                asr_results = session.asr_processor.process_audio_chunk(binary_chunk)

                for asr_item in asr_results:
                    if asr_item.get("type") in ["transcript", "transcript_chunk"]:
                        transcript_text = asr_item.get("text", "")
                        timestamp_ms = asr_item.get("timestamp_ms", 0)
                        
                        # Process complete turn through pipeline
                        analysis_payload = session.process_text_turn(transcript_text, timestamp_ms=timestamp_ms)
                        await websocket.send_json(analysis_payload)
                    else:
                        await websocket.send_json(asr_item)

            # 2. Handle Text Control / Dialogue Frame (JSON)
            elif "text" in message and message["text"]:
                text_content = message["text"]
                try:
                    payload_json = json.loads(text_content)
                    action = payload_json.get("action")

                    if action == "reset":
                        session.reset()
                        await websocket.send_json({"type": "status", "status": "reset"})
                    elif action == "ping":
                        await websocket.send_json({"type": "pong"})
                    elif action == "analyze_text":
                        text_turn = payload_json.get("text", "")
                        analysis_payload = session.process_text_turn(text_turn)
                        await websocket.send_json(analysis_payload)
                    elif action == "flush":
                        flush_results = session.asr_processor.flush()
                        for asr_item in flush_results:
                            if asr_item.get("type") in ["transcript", "transcript_chunk"]:
                                transcript_text = asr_item.get("text", "")
                                analysis_payload = session.process_text_turn(transcript_text)
                                await websocket.send_json(analysis_payload)
                            else:
                                await websocket.send_json(asr_item)
                    else:
                        await websocket.send_json({"type": "info", "message": f"Received action '{action}'"})
                except json.JSONDecodeError:
                    await websocket.send_json({"type": "error", "message": "Invalid JSON text message received."})

    except (WebSocketDisconnect, RuntimeError):
        logger.info("WebSocket client disconnected from ArrestShield stream.")
    except Exception as e:
        logger.error(f"Error handling WebSocket stream: {e}", exc_info=True)
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass
    finally:
        if session is not None:
            session.reset()
