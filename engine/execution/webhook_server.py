"""FastAPI Webhook Server for Angel One Postback Notifications.

Receives real-time order status updates and execution fills pushed by Angel One,
writing notifications to a structured file ledger.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Dict, Any

from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
from logzero import logger
import uvicorn

# Initialize FastAPI App
app = FastAPI(
    title="Chimera Execution Postback Listener",
    description="Receives real-time order fills from Angel One SmartAPI",
)

# Configuration for log storage
LEDGER_DIR = Path("data/execution")
LEDGER_FILE = LEDGER_DIR / "postback_ledger.jsonl"


def append_to_ledger(payload: Dict[str, Any]) -> None:
    """Safely append the postback payload to the local execution ledger file.

    Adds a local receive timestamp for auditability.
    """
    try:
        LEDGER_DIR.mkdir(parents=True, exist_ok=True)
        payload_copy = dict(payload)
        payload_copy["received_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        payload_copy["timestamp_epoch"] = time.time()
        
        with open(LEDGER_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload_copy) + "\n")
            
        logger.info(
            f"Logged postback update | Order ID: {payload.get('orderid')} | "
            f"Status: {payload.get('status')} | Symbol: {payload.get('tradingsymbol')}"
        )
    except Exception as e:
        logger.error(f"Failed to write payload to execution ledger: {e}")


@app.post("/postback")
async def receive_postback(
    request: Request,
    background_tasks: BackgroundTasks,
):
    """Webhook endpoint exposed to Angel One SmartAPI Postback configuration.

    Example payload fields from SmartAPI:
        {
            "clientcode": "S12345",
            "orderid": "240426000000123",
            "tradingsymbol": "RELIANCE-EQ",
            "symboltoken": "2885",
            "transactiontype": "BUY",
            "exchange": "NSE",
            "ordertype": "LIMIT",
            "producttype": "DELIVERY",
            "duration": "DAY",
            "price": "2905.50",
            "quantity": "100",
            "status": "complete",
            "averageprice": "2905.50",
            "filledshares": "100",
            "unfilledshares": "0",
            "orderstatus": "complete",
            "cancelstatus": "false",
            "updateTime": "26-Apr-2024 09:35:12"
        }
    """
    try:
        payload = await request.json()
    except Exception as e:
        logger.error(f"Invalid webhook JSON request: {e}")
        raise HTTPException(status_code=400, detail="Invalid JSON body")
        
    logger.debug(f"Received webhook payload: {payload}")
    
    # Process the update in the background to return HTTP 200 immediately
    background_tasks.add_task(append_to_ledger, payload)
    
    return {"status": "success", "message": "Postback received and queued for ledger log."}


@app.get("/health")
def health_check():
    """Simple status check for process supervisors."""
    return {"status": "ok", "uptime_check": time.time()}


def run_webhook_server(host: str = "0.0.0.0", port: int = 8080) -> None:
    """Helper function to start the webhook server from a daemon process."""
    logger.info(f"Starting postback webhook listener on {host}:{port}...")
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    # If run directly, launch the server
    host_env = os.environ.get("CHIMERA_POSTBACK_HOST", "0.0.0.0")
    port_env = int(os.environ.get("CHIMERA_POSTBACK_PORT", "8080"))
    run_webhook_server(host=host_env, port=port_env)
