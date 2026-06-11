"""
services/ws_manager.py
======================
Manages active WebSocket connections.

The ConnectionManager is a singleton class instantiated at the end of this
module. Both predictions.py (for broadcasting) and the WebSocket endpoint
/ws import it from here — a single source of truth, with no duplication.

Why it lives in services/ instead of routers/:
  The manager is not an endpoint — it is a shared infrastructure component
  used by both the WebSocket endpoint (/ws) and the HTTP endpoint
  (/predict/eta/broadcast).

  If it were defined inside one of the routers, the other router would need
  to import it from there, creating an unnecessary dependency between routers.
  Keeping it in services/ makes it neutral and reusable.
"""

import logging
from fastapi import WebSocket

logger = logging.getLogger(__name__)


class ConnectionManager:
    """
    Maintains the list of connected WebSocket clients and provides
    a method to broadcast a JSON dictionary to all of them.

    Automatic cleanup: if a client disconnects unexpectedly
    (the connection drops without a WebSocketDisconnect event),
    send_json() raises an exception. The manager detects this,
    silently removes the client, and continues sending data
    to the remaining connections.
    """

    def __init__(self):
        self.active: list[WebSocket] = []

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        self.active.append(ws)
        logger.info(
            "WebSocket client connected — total active clients: %d",
            len(self.active),
        )

    def disconnect(self, ws: WebSocket) -> None:
        if ws in self.active:
            self.active.remove(ws)
        logger.info(
            "WebSocket client disconnected — total active clients: %d",
            len(self.active),
        )

    async def broadcast(self, data: dict) -> None:
        """
        Sends `data` as JSON to all connected clients.
        Clients that fail during transmission are removed
        from the active connection list.
        """
        dead: list[WebSocket] = []

        for ws in self.active:
            try:
                await ws.send_json(data)
            except Exception as exc:
                logger.debug(
                    "WebSocket send failed (%s) — client marked for removal.",
                    exc,
                )
                dead.append(ws)

        for ws in dead:
            self.active.remove(ws)

        if dead:
            logger.info(
                "WebSocket broadcast: %d clients removed due to errors — active: %d",
                len(dead),
                len(self.active),
            )


# Singleton — always import from here, never instantiate again.
manager = ConnectionManager()