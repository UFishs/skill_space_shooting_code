"""Websocket RPC server for a hardware-side Gym-like environment."""

from __future__ import annotations

import asyncio
import logging
import traceback

from websockets.legacy.server import serve

from umi.real_world import remote_env_protocol


class WebsocketEnvServer:
    def __init__(self, env, host="0.0.0.0", port=8010, api_key=None):
        self.env = env
        self.host = host
        self.port = port
        self.api_key = api_key
        self._client_active = False
        self._shutdown_event = None
        self._fatal_error = None

    @staticmethod
    def _restart_required(operation, exc):
        """Protocol/operator errors are local; hardware-call failures restart the process."""
        if operation not in {"sample_observation", "reset", "step"}:
            return False
        return not isinstance(exc, (ValueError, PermissionError)) and type(exc).__name__ != "OperatorExit"

    def _dispatch(self, request):
        if self.api_key is not None and request.get("api_key") != self.api_key:
            raise PermissionError("Invalid environment server API key")
        operation = request.get("op")
        if operation == "ping":
            return {"pong": True}
        if operation == "sample_observation":
            return {"observation": self.env.sample_observation()}
        if operation == "reset":
            observation, info = self.env.reset()
            return {"observation": observation, "info": info}
        if operation == "step":
            observation, reward, terminated, truncated, info = self.env.step(request["action"])
            return {
                "observation": observation,
                "reward": reward,
                "terminated": terminated,
                "truncated": truncated,
                "info": info,
            }
        if operation == "end_episode":
            return self.env.end_episode()
        if operation == "close":
            return self.env.end_episode()
        raise ValueError(f"Unknown environment operation: {operation!r}")

    async def _handler(self, websocket):
        if self._client_active:
            await websocket.send(
                remote_env_protocol.packb({"ok": False, "error": "Control server already has a client"})
            )
            await websocket.close()
            return
        self._client_active = True
        try:
            await websocket.send(
                remote_env_protocol.packb({"ok": True, "metadata": self.env.metadata})
            )
            async for packed_request in websocket:
                request = {}
                try:
                    request = remote_env_protocol.unpackb(packed_request)
                    result = self._dispatch(request)
                    response = {"ok": True, "result": result}
                except Exception as exc:
                    logging.exception("Remote environment request failed")
                    restart_required = self._restart_required(request.get("op"), exc)
                    response = {
                        "ok": False,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "traceback": traceback.format_exc(),
                        "restart_required": restart_required,
                    }
                await websocket.send(remote_env_protocol.packb(response))
                if not response.get("ok") and response.get("restart_required"):
                    self._fatal_error = response["error"]
                    self._shutdown_event.set()
                    await websocket.close()
                    return
        finally:
            try:
                self.env.end_episode()
            except Exception:
                logging.exception("Failed to close hardware episode")
            self._client_active = False

    async def _serve_forever(self):
        logging.info("Serving UMI control environment on %s:%d", self.host, self.port)
        self._shutdown_event = asyncio.Event()
        async with serve(self._handler, self.host, self.port, compression=None, max_size=None):
            await self._shutdown_event.wait()

    def serve_forever(self):
        asyncio.run(self._serve_forever())
        if self._fatal_error is not None:
            raise RuntimeError(
                f"Control environment requested a full process restart: {self._fatal_error}"
            )