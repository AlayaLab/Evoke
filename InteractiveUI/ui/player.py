from __future__ import annotations

import asyncio
import contextlib
import json
import math
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

from fastapi import APIRouter, Body, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, RedirectResponse, Response

try:
    from .trajectory import build_pose_npz, validate_camera_samples
except ImportError:
    from trajectory import build_pose_npz, validate_camera_samples

router = APIRouter()
KEYS = frozenset("wasdijkl")
FPS = 30
RECORD_FRAMES = 45


try:
    from .controls import Camera, parse_input
except ImportError:
    from controls import Camera, parse_input


@router.get("/play")
async def player_redirect():
    return RedirectResponse("play/")


@router.get("/play/")
async def player_page():
    return FileResponse(Path(__file__).parent / "static" / "player.html")


@router.post("/api/player/pose")
def export_pose(payload: dict = Body(...)):
    samples = payload.get("cameraSamples")
    try:
        validate_camera_samples(samples, 1)
    except ValueError as error:
        raise HTTPException(400, str(error)) from error
    with tempfile.TemporaryDirectory(prefix="evoke-player-") as directory:
        path = Path(directory) / "pose.npz"
        build_pose_npz(path, [{"x": 0.5, "y": 0.5}], 1, follow_path=False, camera_samples=samples)
        return Response(path.read_bytes(), media_type="application/octet-stream",
                        headers={"Content-Disposition": 'attachment; filename="evoke-player-30fps.npz"'})


@router.websocket("/api/player/ws")
async def player_socket(socket: WebSocket):
    await socket.accept()
    camera = Camera()
    keys: set[str] = set()
    speed, look_speed = 3.0, math.radians(60)
    last_input = time.monotonic()
    recording: list[dict] | None = None
    sequence = 0
    send_lock = asyncio.Lock()

    async def send(data):
        async with send_lock:
            await socket.send_json(data)

    async def receive():
        nonlocal keys, speed, look_speed, last_input, camera, recording
        while True:
            try:
                raw = await socket.receive_text()
                if len(raw) > 4096:
                    await socket.close(code=1009)
                    return
                message = json.loads(raw)
                if not isinstance(message, dict):
                    raise ValueError("Expected an object")
                kind = message.get("type")
                if kind == "input":
                    keys, speed, look_speed = parse_input(message)
                    last_input = time.monotonic()
                elif kind == "reset":
                    camera, keys, recording = Camera(), set(), None
                    await send({"type": "reset"})
                elif kind == "record":
                    if recording is not None:
                        raise ValueError("Recording already in progress")
                    recording = [camera.sample()]
                    await send({"type": "recording", "frames": RECORD_FRAMES, "fps": FPS})
                elif kind == "cancel_record":
                    recording = None
                elif kind == "ping":
                    await send({"type": "pong", "time": message.get("time")})
                else:
                    raise ValueError("Unknown message type")
            except (ValueError, TypeError, OverflowError) as error:
                await send({"type": "error", "message": str(error)})

    async def tick():
        nonlocal sequence, recording
        deadline = time.monotonic()
        while True:
            deadline += 1 / FPS
            await asyncio.sleep(max(0, deadline - time.monotonic()))

            if time.monotonic() - deadline > 1 / FPS:
                deadline = time.monotonic()
            active = keys if time.monotonic() - last_input < 0.4 else set()
            camera.step(active, 1 / FPS, speed, look_speed)
            if recording is not None:
                recording.append(camera.sample())
                if len(recording) == RECORD_FRAMES:
                    await send({"type": "recorded", "samples": recording, "fps": FPS})
                    recording = None
            sequence += 1
            await send({"type": "state", "sequence": sequence, "camera": camera.sample(),
                        "keys": sorted(active), "recordedFrames": len(recording) if recording else 0})

    tasks = [asyncio.create_task(receive()), asyncio.create_task(tick())]
    try:
        done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            task.result()
    except (WebSocketDisconnect, RuntimeError, OSError):
        pass
    finally:
        for task in tasks:
            task.cancel()
        for task in tasks:
            with contextlib.suppress(asyncio.CancelledError, WebSocketDisconnect, RuntimeError, OSError):
                await task
