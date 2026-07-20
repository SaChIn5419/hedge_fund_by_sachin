"""Unit tests for the SmartAPI Execution Engine.

Tests headless authentication, scrip master parsing, routing compliance,
marketable limit calculations, rate limits, and webhook listeners.
"""
from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch
import json
import os
import shutil
import tempfile
import time
from pathlib import Path
import pandas as pd
from fastapi.testclient import TestClient

from engine.execution.engine import SmartAPIExecutionEngine
from engine.execution.webhook_server import app, LEDGER_FILE, append_to_ledger


class TestSmartAPIExecutionEngine(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.engine = SmartAPIExecutionEngine(
            api_key="test_api_key",
            client_code="test_client_code",
            password="test_password",
            totp_secret="MFRGGZDFMZTWQ2LK",  # Valid base32 format
            scrip_cache_dir=self.temp_dir,
            mock_mode=True,  # Test with mock mode by default
        )
        
        # Create dummy scrip master data
        self.dummy_scrips = [
            {"token": "2885", "symbol": "RELIANCE-EQ", "name": "RELIANCE", "exch_seg": "NSE"},
            {"token": "12345", "symbol": "SILVERMIC26APR24", "name": "SILVERMIC", "exch_seg": "MCX"},
            {"token": "9999", "symbol": "NIFTY26APR24FUT", "name": "NIFTY-FUT", "exch_seg": "NFO"},
        ]
        self.engine.scrip_master = pd.DataFrame(self.dummy_scrips)

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_get_token(self):
        token = self.engine.get_token("RELIANCE-EQ", "NSE")
        self.assertEqual(token, "2885")
        
        token = self.engine.get_token("silvermic26apr24", "MCX")
        self.assertEqual(token, "12345")
        
        # Test missing symbol
        token = self.engine.get_token("INVALID", "NSE")
        self.assertIsNone(token)

    @patch("urllib.request.urlopen")
    def test_load_scrip_master_download(self, mock_urlopen):
        # Reset scrip master
        self.engine.scrip_master = None
        
        # Mock response from urlopen
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps(self.dummy_scrips).encode("utf-8")
        mock_urlopen.return_value.__enter__.return_value = mock_response
        
        success = self.engine.load_scrip_master(force_refresh=True)
        self.assertTrue(success)
        self.assertIsNotNone(self.engine.scrip_master)
        self.assertEqual(len(self.engine.scrip_master), 3)
        
        # Check cache file exists
        cache_path = Path(self.temp_dir) / "OpenAPIScripMaster.json"
        self.assertTrue(cache_path.exists())

    def test_marketable_limit_price_calculation(self):
        # We test marketable limit calculation by mocking the LTP response
        self.engine.mock_mode = False
        self.engine.api = MagicMock()
        
        # Mock LTP return value
        self.engine.api.getLtpData.return_value = {
            "status": True,
            "data": {"ltp": "100.0"}
        }
        
        # BUY: price should be LTP * 1.005 = 100.5, rounded to nearest 0.05 tick (100.5)
        buy_price = self.engine.get_marketable_limit_price("RELIANCE-EQ", "NSE", "BUY", buffer_pct=0.005)
        self.assertEqual(buy_price, 100.5)
        
        # SELL: price should be LTP * 0.995 = 99.5
        sell_price = self.engine.get_marketable_limit_price("RELIANCE-EQ", "NSE", "SELL", buffer_pct=0.005)
        self.assertEqual(sell_price, 99.5)
        
        # Test tick rounding: LTP = 100.0, Buy buffer = 0.0016 -> 100.16, should round to 100.15
        buy_price_round = self.engine.get_marketable_limit_price("RELIANCE-EQ", "NSE", "BUY", buffer_pct=0.0016)
        self.assertEqual(buy_price_round, 100.15)

    def test_place_compliant_order_routing(self):
        self.engine.mock_mode = False
        self.engine.api = MagicMock()
        self.engine.api.placeOrder.return_value = "ORD12345"
        
        # Mock ltp retrieval
        self.engine.api.getLtpData.return_value = {
            "status": True,
            "data": {"ltp": "1000.0"}
        }

        # 1. Test Momentum Cash -> DELIVERY segment
        order_id = self.engine.place_compliant_order(
            symbol="RELIANCE-EQ",
            exchange="NSE",
            action="BUY",
            quantity=10,
            strategy_type="MOMENTUM_CASH",
            limit_price=1005.00
        )
        self.assertEqual(order_id, "ORD12345")
        
        # Verify variety, ordertype, duration compliance
        call_args = self.engine.api.placeOrder.call_args[0][0]
        self.assertEqual(call_args["variety"], "NORMAL")
        self.assertEqual(call_args["ordertype"], "LIMIT")
        self.assertEqual(call_args["duration"], "DAY")
        self.assertEqual(call_args["producttype"], "DELIVERY")
        self.assertEqual(call_args["price"], "1005.00")
        self.assertEqual(call_args["quantity"], "10")

        # 2. Test Mean Reversion Deriv -> CARRYFORWARD segment
        order_id_mr = self.engine.place_compliant_order(
            symbol="SILVERMIC26APR24",
            exchange="MCX",
            action="SELL",
            quantity=5,
            strategy_type="MEAN_REV_DERIV",
            limit_price=74000.00
        )
        self.assertEqual(order_id_mr, "ORD12345")
        call_args_mr = self.engine.api.placeOrder.call_args_list[-1][0][0]
        self.assertEqual(call_args_mr["producttype"], "CARRYFORWARD")
        self.assertEqual(call_args_mr["price"], "74000.00")

    def test_rate_limiter_order_spacing(self):
        # We test that spacing is enforced when calling rate limiter in sequence
        start = time.time()
        self.engine.enforce_rate_limit(is_poll=False)
        self.engine.enforce_rate_limit(is_poll=False)
        elapsed = time.time() - start
        
        # Elapsed time should be at least 100ms (self.order_spacing)
        self.assertGreaterEqual(elapsed, 0.08)  # Allow small buffer for timer jitter

    def test_rate_limiter_poll_spacing(self):
        start = time.time()
        self.engine.enforce_rate_limit(is_poll=True, poll_endpoint="test_poll")
        self.engine.enforce_rate_limit(is_poll=True, poll_endpoint="test_poll")
        elapsed = time.time() - start
        
        # Elapsed time should be at least 1.0s (self.poll_spacing)
        self.assertGreaterEqual(elapsed, 0.9)


class TestWebhookServer(unittest.TestCase):

    def setUp(self):
        self.client = TestClient(app)
        # Clear ledger if exists
        if LEDGER_FILE.exists():
            os.remove(LEDGER_FILE)

    def tearDown(self):
        if LEDGER_FILE.exists():
            os.remove(LEDGER_FILE)

    def test_health_check(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "ok")

    def test_postback_webhook_receiver(self):
        payload = {
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
        
        # POST to webhook
        response = self.client.post("/postback", json=payload)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "success")
        
        # Sleep briefly to let BackgroundTasks finish writing to file
        time.sleep(0.1)
        
        # Check that file exists and has content
        self.assertTrue(LEDGER_FILE.exists())
        with open(LEDGER_FILE, "r", encoding="utf-8") as f:
            lines = f.readlines()
            
        self.assertEqual(len(lines), 1)
        logged_data = json.loads(lines[0])
        self.assertEqual(logged_data["orderid"], "240426000000123")
        self.assertEqual(logged_data["tradingsymbol"], "RELIANCE-EQ")
        self.assertIn("received_at", logged_data)


if __name__ == "__main__":
    unittest.main()
