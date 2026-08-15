import json
import logging
from typing import Optional, Any
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from src.asr.streaming import StreamingASRProcessor
from src.config import settings

logger = logging.getLogger("ArrestShield.WebSocket")
router = APIRouter()


@router.websocket(settings.server.ws_asr_path)
async def websocket_asr_endpoint(
    websocket: WebSocket,
    processor: Optional[Any] = None
):
    """
    Streaming WebSocket handler for real-time speech-to-text.
    
    Accepts binary PCM audio frames from client stream, runs VAD,
    transcribes speech incrementally under 800ms latency, and yields JSON text segments.
    """
    await websocket.accept()
    logger.info(f"WebSocket client connected to streaming ASR endpoint '{settings.server.ws_asr_path}'")

    if processor is None:
        try:
            processor = StreamingASRProcessor(
                sample_rate=settings.asr.sample_rate,
                window_duration_sec=settings.asr.window_duration_sec,
                latency_target_ms=settings.asr.latency_target_ms
            )
        except Exception as e:
            logger.error(f"Failed to initialize StreamingASRProcessor for WebSocket: {e}")
            await websocket.send_json({
                "type": "error",
                "message": f"Initialization error: {str(e)}"
            })
            await websocket.close(code=1011)
            return

    try:
        while True:
            message = await websocket.receive()
            
            # 1. Handle Binary Audio Frame
            if "bytes" in message and message["bytes"]:
                binary_chunk = message["bytes"]
                results = processor.process_audio_chunk(binary_chunk)
                for payload in results:
                    await websocket.send_json(payload)

            # 2. Handle Text Control Frame (JSON)
            elif "text" in message and message["text"]:
                text_content = message["text"]
                try:
                    payload_json = json.loads(text_content)
                    action = payload_json.get("action")

                    if action == "reset":
                        processor.reset()
                        await websocket.send_json({"type": "status", "status": "reset"})
                    elif action == "ping":
                        await websocket.send_json({"type": "pong"})
                    elif action == "flush":
                        flush_results = processor.flush()
                        for payload in flush_results:
                            await websocket.send_json(payload)
                    else:
                        await websocket.send_json({
                            "type": "info",
                            "message": f"Received action '{action}'"
                        })
                except json.JSONDecodeError:
                    await websocket.send_json({
                        "type": "error",
                        "message": "Invalid JSON text message received."
                    })

    except (WebSocketDisconnect, RuntimeError):
        logger.info("WebSocket client disconnected from ASR stream.")
    except Exception as e:
        logger.error(f"Error handling ASR WebSocket stream: {e}", exc_info=True)
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass
    finally:
        if processor is not None:
            processor.reset()

