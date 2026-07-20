"""SmartAPI Execution Engine for Angel One.

Handles headless authentication (TOTP), scrip master caching, multi-segment
execution routing, marketable limit order translation, compliance checks, and rate limiting.
"""
from __future__ import annotations

import datetime
import json
import os
import time
import urllib.request
from pathlib import Path
from typing import Dict, Any, Optional

import pandas as pd
from logzero import logger
import pyotp

# Try importing SmartConnect, fall back to mock if not installed (for testing/non-live)
try:
    from SmartApi import SmartConnect
except ImportError:
    SmartConnect = None
    logger.warning("smartapi-python is not installed. Live operations will be mock-only until installed.")


class SmartAPIExecutionEngine:
    """Production-grade Execution Engine for Angel One's SmartAPI."""

    def __init__(
        self,
        api_key: str,
        client_code: str,
        password: str,
        totp_secret: str,
        scrip_cache_dir: str = "data",
        mock_mode: bool = False,
    ):
        self.api_key = api_key
        self.client_code = client_code
        self.password = password
        self.totp_secret = totp_secret
        self.scrip_cache_dir = Path(scrip_cache_dir)
        self.mock_mode = mock_mode or (SmartConnect is None)
        
        self.api: Optional[Any] = None
        self.scrip_master: Optional[pd.DataFrame] = None
        self.last_scrip_update: Optional[datetime.date] = None
        
        # Rate Limiting State (10 Orders Per Second -> 100ms spacing)
        self.last_order_time = 0.0
        self.order_spacing = 0.1  # seconds
        
        # Rate Limiting Polling (1 Request Per Second for getOrderBook etc)
        self.last_poll_times: Dict[str, float] = {}
        self.poll_spacing = 1.0  # seconds

    def initialize_session(self) -> bool:
        """Authenticate with SmartAPI using TOTP automation."""
        if self.mock_mode:
            logger.info("[MOCK] Initialized mock SmartAPI execution session.")
            return True

        if SmartConnect is None:
            logger.error("SmartConnect is not available. Install smartapi-python to run live.")
            return False

        logger.info("Initializing session with SmartAPI execution engine...")
        self.api = SmartConnect(api_key=self.api_key)
        
        # Generate current TOTP from the base32 secret
        try:
            totp = pyotp.TOTP(self.totp_secret).now()
        except Exception as e:
            logger.error(f"Failed to generate TOTP: {e}")
            return False

        try:
            session = self.api.generateSession(self.client_code, self.password, totp)
            if session.get("status"):
                logger.info("Fund successfully connected to SmartAPI execution engine.")
                return True
            else:
                logger.error(f"Authentication Failed: {session.get('message')}")
                return False
        except Exception as e:
            logger.exception(f"Connection Exception: {e}")
            return False

    def load_scrip_master(self, force_refresh: bool = False) -> bool:
        """Fetch and cache the Angel One Scrip Master locally to avoid daily redownloads."""
        self.scrip_cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = self.scrip_cache_dir / "OpenAPIScripMaster.json"
        
        today = datetime.date.today()
        cache_valid = False
        
        if cache_path.exists() and not force_refresh:
            # Check if file was modified today
            mtime = datetime.date.fromtimestamp(cache_path.stat().st_mtime)
            if mtime == today:
                cache_valid = True
                
        if cache_valid:
            logger.info("Loading Scrip Master from local cache...")
            try:
                self.scrip_master = pd.read_json(cache_path)
                self.last_scrip_update = today
                logger.info(f"Loaded {len(self.scrip_master)} scrips from local cache.")
                return True
            except Exception as e:
                logger.warning(f"Failed to read cached Scrip Master: {e}. Re-downloading...")

        # Re-download Scrip Master
        url = "https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json"
        logger.info(f"Downloading Scrip Master from {url}...")
        try:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
            )
            with urllib.request.urlopen(req) as response:
                data = json.loads(response.read().decode("utf-8"))
                
            # Write to cache
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(data, f)
                
            self.scrip_master = pd.DataFrame(data)
            self.last_scrip_update = today
            logger.info(f"Downloaded and cached {len(self.scrip_master)} scrips.")
            return True
        except Exception as e:
            logger.error(f"Failed to download Scrip Master: {e}")
            return False

    def get_token(self, symbol: str, exchange: str) -> Optional[str]:
        """Retrieve the unique symboltoken required for order placement."""
        if self.scrip_master is None:
            logger.error("Scrip Master not loaded. Call load_scrip_master() first.")
            return None
            
        # Clean symbol to match master format (e.g. UPPERCASE)
        symbol_upper = symbol.strip().upper()
        exch_upper = exchange.strip().upper()
        
        # Search for exact symbol match in exchange
        matches = self.scrip_master[
            (self.scrip_master["symbol"] == symbol_upper) &
            (self.scrip_master["exch_seg"] == exch_upper)
        ]
        
        if not matches.empty:
            return str(matches.iloc[0]["token"])
            
        logger.warning(f"SymbolToken not found in Scrip Master for {symbol_upper} on {exch_upper}")
        return None

    def enforce_rate_limit(self, is_poll: bool = False, poll_endpoint: str = "") -> None:
        """Enforces rate limits: 10 orders/sec globally, 1 request/sec for polling endpoints."""
        now = time.time()
        
        if is_poll and poll_endpoint:
            last_poll = self.last_poll_times.get(poll_endpoint, 0.0)
            elapsed = now - last_poll
            if elapsed < self.poll_spacing:
                wait_time = self.poll_spacing - elapsed
                logger.debug(f"Rate Limiting poll for {poll_endpoint}: sleeping {wait_time:.2f}s")
                time.sleep(wait_time)
                now = time.time()
            self.last_poll_times[poll_endpoint] = now
        else:
            elapsed = now - self.last_order_time
            if elapsed < self.order_spacing:
                wait_time = self.order_spacing - elapsed
                logger.debug(f"Rate Limiting order placement: sleeping {wait_time:.2f}s")
                time.sleep(wait_time)
                now = time.time()
            self.last_order_time = now

    def get_marketable_limit_price(
        self,
        symbol: str,
        exchange: str,
        action: str,
        buffer_pct: float = 0.005,  # 0.5% default buffer
    ) -> float:
        """Fetch Last Traded Price (LTP) and compute marketable limit price.

        Buy orders: LTP * (1 + buffer_pct)
        Sell orders: LTP * (1 - buffer_pct)
        """
        if self.mock_mode:
            # Return dummy price for testing
            return 100.0

        token = self.get_token(symbol, exchange)
        if not token:
            raise ValueError(f"Cannot get LTP, token not found for {symbol} on {exchange}")
            
        self.enforce_rate_limit(is_poll=True, poll_endpoint="getLtpData")
        
        try:
            res = self.api.getLtpData(exchange=exchange, tradingsymbol=symbol, symboltoken=token)
            if res.get("status") and "data" in res and "ltp" in res["data"]:
                ltp = float(res["data"]["ltp"])
                if action.upper() == "BUY":
                    price = ltp * (1.0 + buffer_pct)
                else:
                    price = ltp * (1.0 - buffer_pct)
                
                # Round to nearest tick (0.05 paisa for NSE)
                price = round(price * 20.0) / 20.0
                logger.info(f"LTP for {symbol} is {ltp}. Adjusted marketable limit price to {price} ({action})")
                return price
            else:
                logger.error(f"Failed to fetch LTP: {res.get('message')}")
                raise ValueError("Failed to fetch LTP from API response")
        except Exception as e:
            logger.exception(f"Exception fetching LTP: {e}")
            raise

    def place_compliant_order(
        self,
        symbol: str,
        exchange: str,
        action: str,
        quantity: int,
        strategy_type: str,
        limit_price: Optional[float] = None,
        marketable_buffer_pct: float = 0.005,
    ) -> Optional[str]:
        """Routes compliance-conforming LIMIT orders (Market/IOC banned under 2026 guidelines).

        Parameters
        ----------
        symbol : str
            Trading symbol, e.g., "RELIANCE-EQ" or "SILVERMIC26APR24".
        exchange : str
            Exchange segment, e.g., "NSE", "BSE", "NFO", "MCX".
        action : str
            "BUY" or "SELL".
        quantity : int
            Quantity to trade.
        strategy_type : str
            "MOMENTUM_CASH" or "MEAN_REV_DERIV".
        limit_price : Optional[float], optional
            Specify exact limit price. If None, retrieves LTP and calculates
            marketable limit price for immediate fill.
        marketable_buffer_pct : float, optional
            Buffer percentage for marketable limit order. Default is 0.5%.

        Returns
        -------
        order_id : Optional[str]
            Placed Order ID if successful, None otherwise.
        """
        token = self.get_token(symbol, exchange)
        if not token:
            logger.error(f"Cannot place order: SymbolToken not found for {symbol}")
            return None
            
        action_upper = action.strip().upper()
        if action_upper not in ["BUY", "SELL"]:
            logger.error(f"Invalid action: {action_upper}")
            return None

        # 1. Map segment product type
        if strategy_type == "MOMENTUM_CASH":
            prod_type = "DELIVERY"
        elif strategy_type == "MEAN_REV_DERIV":
            prod_type = "CARRYFORWARD"
        else:
            prod_type = "INTRADAY"

        # 2. Determine compliance limit price
        final_price = limit_price
        if final_price is None:
            try:
                final_price = self.get_marketable_limit_price(
                    symbol=symbol,
                    exchange=exchange,
                    action=action_upper,
                    buffer_pct=marketable_buffer_pct
                )
            except Exception as e:
                logger.error(f"Could not compute marketable limit price: {e}. Order aborted.")
                return None

        # 3. Create compliant order parameters
        order_params = {
            "variety": "NORMAL",
            "tradingsymbol": symbol.upper(),
            "symboltoken": token,
            "transactiontype": action_upper,
            "exchange": exchange.upper(),
            "ordertype": "LIMIT",      # Compliance: strictly LIMIT (No Market)
            "producttype": prod_type,
            "duration": "DAY",         # Compliance: strictly DAY (No IOC)
            "price": f"{final_price:.2f}",
            "quantity": str(quantity),
            "squareoff": "0",
            "stoploss": "0"
        }

        # 4. Enforce order rate limiting (10 OPS)
        self.enforce_rate_limit(is_poll=False)

        if self.mock_mode:
            mock_order_id = f"MOCK_ORD_{int(time.time() * 1000)}"
            logger.info(
                f"[MOCK] Placed compliant order | ID: {mock_order_id} | "
                f"{action_upper} {quantity} {symbol} @ {final_price:.2f} | Prod: {prod_type}"
            )
            return mock_order_id

        try:
            order_id = self.api.placeOrder(order_params)
            logger.info(
                f"Order Placed | ID: {order_id} | "
                f"{action_upper} {quantity} {symbol} @ {final_price:.2f} | Prod: {prod_type}"
            )
            return order_id
        except Exception as e:
            logger.exception(f"Execution Engine Order Placement Failure: {e}")
            return None

    def get_order_book(self) -> Optional[list]:
        """Fetch order book with polling rate limiting."""
        if self.mock_mode:
            logger.info("[MOCK] Fetching mock order book (empty)")
            return []
            
        self.enforce_rate_limit(is_poll=True, poll_endpoint="getOrderBook")
        try:
            res = self.api.orderBook()
            if res.get("status"):
                return res.get("data", [])
            else:
                logger.error(f"Failed to fetch order book: {res.get('message')}")
                return None
        except Exception as e:
            logger.exception(f"Exception fetching order book: {e}")
            return None
