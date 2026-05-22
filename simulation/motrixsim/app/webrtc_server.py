# SPDX-FileCopyrightText: Copyright (c) MOS-Brain Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""WebRTC H.264 preview (aiortc): lower CPU / bandwidth than per-frame JPEG over Socket.IO.

Requires: pip install aiortc aiohttp
Encoding uses FFmpeg inside PyAV (aiortc); NVIDIA NVENC is used when the FFmpeg build supports it.

Signaling: POST /webrtc/offer with JSON {\"sdp\",\"type\"} from the browser; returns answer SDP.
"""

from __future__ import annotations

import asyncio
import threading
import time
import traceback
from fractions import Fraction
from typing import Any

_have_aiortc = False
try:
    from aiohttp import web
    from aiortc import (
        RTCConfiguration,
        RTCIceServer,
        RTCPeerConnection,
        RTCSessionDescription,
        VideoStreamTrack,
    )
    from av import VideoFrame

    _have_aiortc = True
except ImportError:
    web = None  # type: ignore[misc,assignment]
    VideoFrame = None  # type: ignore[misc,assignment]
    VideoStreamTrack = object  # type: ignore[misc,assignment]


def webrtc_dependencies_available() -> bool:
    return bool(_have_aiortc)


if _have_aiortc:

    class SimulationVideoTrack(VideoStreamTrack):  # type: ignore[misc,valid-type]
        kind = "video"

        def __init__(self, fps: float) -> None:
            super().__init__()
            self._fps = max(1.0, float(fps))
            self._queue: asyncio.Queue = asyncio.Queue(maxsize=2)
            self._t0: float | None = None
            self._loop: asyncio.AbstractEventLoop | None = None

        def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
            self._loop = loop

        def push_rgb(self, rgb: np.ndarray) -> None:
            loop = self._loop
            if loop is None or loop.is_closed():
                return
            if rgb.dtype != np.uint8:
                rgb = np.clip(rgb, 0, 255).astype(np.uint8)
            if rgb.ndim == 3 and rgb.shape[2] == 4:
                rgb = rgb[:, :, :3]
            rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
            asyncio.run_coroutine_threadsafe(self._enqueue(rgb), loop)

        async def _enqueue(self, rgb: np.ndarray) -> None:
            if self._queue.full():
                try:
                    self._queue.get_nowait()
                except asyncio.QueueEmpty:
                    pass
            await self._queue.put(rgb)

        async def recv(self) -> Any:
            if self._t0 is None:
                self._t0 = time.time()
            rgb = await self._queue.get()
            frame = VideoFrame.from_ndarray(rgb, format="rgb24")
            frame.pts = int((time.time() - self._t0) * 90000)
            frame.time_base = Fraction(1, 90000)
            return frame

else:

    class SimulationVideoTrack:  # type: ignore[no-redef]
        def __init__(self, fps: float) -> None:
            raise RuntimeError("aiortc not installed")


if _have_aiortc:

    @web.middleware
    async def _cors_middleware(request: web.Request, handler: Any) -> web.StreamResponse:
        if request.method == "OPTIONS":
            return web.Response(
                status=204,
                headers={
                    "Access-Control-Allow-Origin": "*",
                    "Access-Control-Allow-Methods": "POST, OPTIONS",
                    "Access-Control-Allow-Headers": "Content-Type",
                },
            )
        resp = await handler(request)
        resp.headers["Access-Control-Allow-Origin"] = "*"
        return resp


async def _webrtc_app_main(
    host: str,
    port: int,
    rgb_q: Any,
    fps: int,
    stop: threading.Event,
    active_track: dict[str, SimulationVideoTrack | None],
    pcs: set,
) -> None:
    from aiohttp import web
    from aiortc import (
        RTCConfiguration,
        RTCIceServer,
        RTCPeerConnection,
        RTCSessionDescription,
    )

    lock_pc = asyncio.Lock()

    async def offer(request: web.Request) -> web.Response:
        try:
            params = await request.json()
        except Exception as e:
            return web.json_response({"error": f"invalid json: {e}"}, status=400)
        if not isinstance(params, dict) or "sdp" not in params or "type" not in params:
            return web.json_response({"error": "expected sdp and type"}, status=400)

        offer_desc = RTCSessionDescription(sdp=params["sdp"], type=params["type"])
        configuration = RTCConfiguration(
            iceServers=[RTCIceServer(urls=["stun:stun.l.google.com:19302"])]
        )
        peer = RTCPeerConnection(configuration=configuration)

        track = SimulationVideoTrack(float(fps))
        track.bind_loop(asyncio.get_running_loop())

        async with lock_pc:
            for old in list(pcs):
                try:
                    await old.close()
                except Exception:
                    pass
            pcs.clear()
            active_track["track"] = track
            pcs.add(peer)

        @peer.on("connectionstatechange")
        async def on_state() -> None:
            st = peer.connectionState
            print(f"[MotrixWebRTC] PeerConnection state={st}", flush=True)
            if st in ("failed", "closed", "disconnected"):
                try:
                    await peer.close()
                except Exception:
                    pass
                pcs.discard(peer)
                if active_track.get("track") is track:
                    active_track["track"] = None

        try:
            await peer.setRemoteDescription(offer_desc)
            peer.addTrack(track)
            answer = await peer.createAnswer()
            await peer.setLocalDescription(answer)
        except Exception as e:
            traceback.print_exc()
            pcs.discard(peer)
            if active_track.get("track") is track:
                active_track["track"] = None
            try:
                await peer.close()
            except Exception:
                pass
            return web.json_response({"error": str(e)}, status=500)

        return web.json_response(
            {"sdp": peer.localDescription.sdp, "type": peer.localDescription.type}
        )

    async def on_options(_request: web.Request) -> web.Response:
        return web.Response(
            status=204,
            headers={
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "POST, OPTIONS",
                "Access-Control-Allow-Headers": "Content-Type",
            },
        )

    app = web.Application(middlewares=[_cors_middleware])
    app.router.add_post("/webrtc/offer", offer)
    app.router.add_options("/webrtc/offer", on_options)

    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, host, port)
    await site.start()
    print(
        f"[MotrixWebRTC] signaling at http://{host}:{port}/webrtc/offer (POST JSON offer)",
        flush=True,
    )

    def rgb_pump() -> None:
        while not stop.is_set():
            try:
                rgb = rgb_q.get(timeout=0.05)
            except Exception:
                continue
            if rgb is None:
                break
            t = active_track.get("track")
            if t is not None:
                try:
                    t.push_rgb(rgb)
                except Exception:
                    pass

    threading.Thread(target=rgb_pump, name="motrix-webrtc-rgb", daemon=True).start()

    try:
        while not stop.is_set():
            await asyncio.sleep(0.25)
    finally:
        for pc in list(pcs):
            try:
                await pc.close()
            except Exception:
                pass
        pcs.clear()
        active_track["track"] = None
        await runner.cleanup()


def start_webrtc_server_thread(
    host: str,
    port: int,
    rgb_q: Any,
    fps: int,
    stop: threading.Event,
) -> bool:
    if not webrtc_dependencies_available():
        print(
            "[MotrixWebRTC] missing deps; install: pip install aiortc aiohttp",
            flush=True,
        )
        return False

    active_track: dict[str, SimulationVideoTrack | None] = {"track": None}
    pcs: set = set()

    def runner() -> None:
        try:
            asyncio.run(_webrtc_app_main(host, port, rgb_q, fps, stop, active_track, pcs))
        except OSError as e:
            print(f"[MotrixWebRTC] server failed: {e}", flush=True)
        except Exception:
            traceback.print_exc()

    threading.Thread(target=runner, name="motrix-webrtc-aiohttp", daemon=True).start()
    return True
