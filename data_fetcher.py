"""
data_fetcher.py — Binance API OHLCV fetcher
Obtiene datos históricos y en tiempo real de Bitcoin desde Binance
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time

BINANCE_BASE = "https://api.binance.com/api/v3"

INTERVALS = {
    "1h":  "1h",
    "4h":  "4h",
    "1d":  "1d",
    "15m": "15m",
}

def fetch_ohlcv(symbol: str = "BTCUSDT", interval: str = "1h", limit: int = 500) -> pd.DataFrame:
    """
    Descarga datos OHLCV desde Binance (sin API key — endpoint público).
    
    Args:
        symbol:   Par de trading, ej. 'BTCUSDT'
        interval: '15m', '1h', '4h', '1d'
        limit:    Número de velas (máx 1000)
    
    Returns:
        DataFrame con columnas OHLCV + timestamp
    """
    url = f"{BINANCE_BASE}/klines"
    params = {
        "symbol":   symbol,
        "interval": interval,
        "limit":    min(limit, 1000),
    }

    resp = requests.get(url, params=params, timeout=10)
    resp.raise_for_status()
    raw = resp.json()

    df = pd.DataFrame(raw, columns=[
        "timestamp", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trades",
        "taker_buy_base", "taker_buy_quote", "ignore"
    ])

    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    for col in ["open", "high", "low", "close", "volume", "quote_volume"]:
        df[col] = df[col].astype(float)

    df.set_index("timestamp", inplace=True)
    df = df[["open", "high", "low", "close", "volume", "quote_volume", "trades"]]
    df["trades"] = df["trades"].astype(int)

    return df


def fetch_current_price(symbol: str = "BTCUSDT") -> dict:
    """Precio actual + cambio 24h."""
    url = f"{BINANCE_BASE}/ticker/24hr"
    resp = requests.get(url, params={"symbol": symbol}, timeout=10)
    resp.raise_for_status()
    d = resp.json()
    return {
        "price":         float(d["lastPrice"]),
        "change_24h":    float(d["priceChangePercent"]),
        "high_24h":      float(d["highPrice"]),
        "low_24h":       float(d["lowPrice"]),
        "volume_24h":    float(d["volume"]),
        "trades_24h":    int(d["count"]),
    }


def fetch_order_book_depth(symbol: str = "BTCUSDT", limit: int = 20) -> dict:
    """Profundidad del libro de órdenes para detectar soporte/resistencia."""
    url = f"{BINANCE_BASE}/depth"
    resp = requests.get(url, params={"symbol": symbol, "limit": limit}, timeout=10)
    resp.raise_for_status()
    ob = resp.json()

    bids = pd.DataFrame(ob["bids"], columns=["price", "qty"], dtype=float)
    asks = pd.DataFrame(ob["asks"], columns=["price", "qty"], dtype=float)

    bid_wall = bids.loc[bids["qty"].idxmax()]
    ask_wall = asks.loc[asks["qty"].idxmax()]

    return {
        "best_bid":      bids["price"].max(),
        "best_ask":      asks["price"].min(),
        "spread":        asks["price"].min() - bids["price"].max(),
        "bid_volume":    bids["qty"].sum(),
        "ask_volume":    asks["qty"].sum(),
        "buy_pressure":  bids["qty"].sum() / (bids["qty"].sum() + asks["qty"].sum()),
        "bid_wall_price": float(bid_wall["price"]),
        "ask_wall_price": float(ask_wall["price"]),
    }


def fetch_multi_timeframe(symbol: str = "BTCUSDT") -> dict:
    """Descarga datos en los 3 timeframes principales de una vez."""
    print(f"📡 Descargando datos multi-timeframe para {symbol}...")
    data = {}
    for tf in ["1h", "4h", "1d"]:
        limit = 500 if tf == "1h" else 300
        data[tf] = fetch_ohlcv(symbol, interval=tf, limit=limit)
        print(f"   ✅ {tf}: {len(data[tf])} velas cargadas")
        time.sleep(0.2)
    return data


if __name__ == "__main__":
    print("=== BINANCE DATA FETCHER TEST ===\n")
    
    price_info = fetch_current_price()
    print(f"💰 Bitcoin Precio Actual: ${price_info['price']:,.2f}")
    print(f"   Cambio 24h: {price_info['change_24h']:+.2f}%")
    print(f"   High 24h:   ${price_info['high_24h']:,.2f}")
    print(f"   Low 24h:    ${price_info['low_24h']:,.2f}")
    print(f"   Volumen 24h: {price_info['volume_24h']:,.0f} BTC\n")

    ob = fetch_order_book_depth()
    print(f"📊 Order Book:")
    print(f"   Spread:       ${ob['spread']:.2f}")
    print(f"   Buy Pressure: {ob['buy_pressure']*100:.1f}%")
    print(f"   Bid Wall:     ${ob['bid_wall_price']:,.0f}")
    print(f"   Ask Wall:     ${ob['ask_wall_price']:,.0f}\n")

    df = fetch_ohlcv(interval="1h", limit=10)
    print(f"📈 Últimas 10 velas (1h):\n{df[['open','high','low','close','volume']].tail(10)}")
