"""
features.py — Feature Engineering profesional
RSI, MACD, Bollinger Bands, ATR, EMA stack, Volume features,
Momentum, Fractal patterns y más.
"""

import pandas as pd
import numpy as np


# ─── Indicadores base ────────────────────────────────────────────────────────

def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain  = delta.clip(lower=0).ewm(com=period - 1, adjust=False).mean()
    loss  = (-delta.clip(upper=0)).ewm(com=period - 1, adjust=False).mean()
    rs    = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def macd(close: pd.Series, fast=12, slow=26, signal=9) -> pd.DataFrame:
    ema_fast   = close.ewm(span=fast, adjust=False).mean()
    ema_slow   = close.ewm(span=slow, adjust=False).mean()
    macd_line  = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram  = macd_line - signal_line
    return pd.DataFrame({
        "macd":      macd_line,
        "macd_sig":  signal_line,
        "macd_hist": histogram,
    })


def bollinger_bands(close: pd.Series, period: int = 20, std: float = 2.0) -> pd.DataFrame:
    ma    = close.rolling(period).mean()
    sigma = close.rolling(period).std()
    upper = ma + std * sigma
    lower = ma - std * sigma
    return pd.DataFrame({
        "bb_upper":  upper,
        "bb_mid":    ma,
        "bb_lower":  lower,
        "bb_width":  (upper - lower) / ma,       # volatilidad normalizada
        "bb_pct":    (close - lower) / (upper - lower),  # posición dentro de BB
    })


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low  - close.shift()).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(com=period - 1, adjust=False).mean()


def stochastic(high, low, close, k_period=14, d_period=3) -> pd.DataFrame:
    lowest_low   = low.rolling(k_period).min()
    highest_high = high.rolling(k_period).max()
    k = 100 * (close - lowest_low) / (highest_high - lowest_low)
    d = k.rolling(d_period).mean()
    return pd.DataFrame({"stoch_k": k, "stoch_d": d})


def cci(high, low, close, period=20) -> pd.Series:
    tp   = (high + low + close) / 3
    ma   = tp.rolling(period).mean()
    mad  = tp.rolling(period).apply(lambda x: np.mean(np.abs(x - x.mean())))
    return (tp - ma) / (0.015 * mad)


def williams_r(high, low, close, period=14) -> pd.Series:
    highest_high = high.rolling(period).max()
    lowest_low   = low.rolling(period).min()
    return -100 * (highest_high - close) / (highest_high - lowest_low)


def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    direction = np.sign(close.diff()).fillna(0)
    return (direction * volume).cumsum()


def vwap(high, low, close, volume) -> pd.Series:
    tp = (high + low + close) / 3
    return (tp * volume).cumsum() / volume.cumsum()


# ─── Feature Engineering completo ────────────────────────────────────────────

def build_features(df: pd.DataFrame, target_horizon: int = 1) -> pd.DataFrame:
    """
    Construye el dataset completo de features + target para ML.
    
    Args:
        df:               DataFrame con columnas OHLCV
        target_horizon:   Número de velas adelante para el target (default 1)
    
    Returns:
        DataFrame con todas las features y columna 'target'
    """
    f = df.copy()
    c = f["close"]
    h = f["high"]
    l = f["low"]
    v = f["volume"]

    # ── Price features ────────────────────────────────────────────────────────
    for p in [5, 10, 20, 50, 100, 200]:
        f[f"ema_{p}"]    = c.ewm(span=p, adjust=False).mean()
        f[f"sma_{p}"]    = c.rolling(p).mean()
        f[f"ret_{p}"]    = c.pct_change(p)           # retorno p velas
        f[f"dist_ema_{p}"] = (c - f[f"ema_{p}"]) / f[f"ema_{p}"]  # distancia % a EMA

    # ── Momentum ──────────────────────────────────────────────────────────────
    f["rsi_14"]   = rsi(c, 14)
    f["rsi_7"]    = rsi(c, 7)
    f["rsi_21"]   = rsi(c, 21)
    f["rsi_div"]  = f["rsi_14"] - f["rsi_21"]        # divergencia RSI rápido/lento

    macd_df = macd(c)
    f = pd.concat([f, macd_df], axis=1)

    stoch_df = stochastic(h, l, c)
    f = pd.concat([f, stoch_df], axis=1)

    f["cci_20"]    = cci(h, l, c, 20)
    f["williams_r"] = williams_r(h, l, c, 14)

    # ── Volatilidad ───────────────────────────────────────────────────────────
    bb_df = bollinger_bands(c)
    f = pd.concat([f, bb_df], axis=1)

    f["atr_14"]     = atr(h, l, c, 14)
    f["atr_pct"]    = f["atr_14"] / c               # ATR normalizado
    f["volatility_20"] = c.pct_change().rolling(20).std() * np.sqrt(252)  # vol anualizada
    f["volatility_5"]  = c.pct_change().rolling(5).std()

    # ── Candle patterns ───────────────────────────────────────────────────────
    f["body"]         = (c - f["open"]).abs() / f["open"]
    f["upper_shadow"] = (h - c.clip(lower=f["open"])) / f["open"]
    f["lower_shadow"] = (c.clip(upper=f["open"]) - l) / f["open"]
    f["candle_dir"]   = np.sign(c - f["open"])         # 1=bullish, -1=bearish
    f["doji"]         = (f["body"] < 0.001).astype(int)
    f["high_low_pct"] = (h - l) / l

    # ── Volume features ───────────────────────────────────────────────────────
    f["volume_sma20"]   = v.rolling(20).mean()
    f["volume_ratio"]   = v / f["volume_sma20"]        # volumen relativo
    f["obv"]            = obv(c, v)
    f["obv_ema"]        = f["obv"].ewm(span=20).mean()
    f["obv_signal"]     = f["obv"] - f["obv_ema"]
    f["vwap"]           = vwap(h, l, c, v)
    f["dist_vwap"]      = (c - f["vwap"]) / f["vwap"]
    f["volume_momentum"] = v.pct_change(5)

    # ── Trend features ────────────────────────────────────────────────────────
    f["trend_20_50"]  = (f["ema_20"] > f["ema_50"]).astype(int)
    f["trend_50_200"] = (f["ema_50"] > f["ema_200"]).astype(int)
    f["golden_cross"] = ((f["ema_20"] > f["ema_50"]) & 
                          (f["ema_20"].shift() <= f["ema_50"].shift())).astype(int)
    f["death_cross"]  = ((f["ema_20"] < f["ema_50"]) & 
                          (f["ema_20"].shift() >= f["ema_50"].shift())).astype(int)

    # ── Support / Resistance ──────────────────────────────────────────────────
    f["rolling_high_20"] = h.rolling(20).max()
    f["rolling_low_20"]  = l.rolling(20).min()
    f["dist_high_20"]    = (c - f["rolling_high_20"]) / f["rolling_high_20"]
    f["dist_low_20"]     = (c - f["rolling_low_20"])  / f["rolling_low_20"]

    # ── Lag features (pasado reciente) ────────────────────────────────────────
    for lag in [1, 2, 3, 5, 8]:
        f[f"return_lag_{lag}"] = c.pct_change().shift(lag)
        f[f"vol_lag_{lag}"]    = f["volume_ratio"].shift(lag)
        f[f"rsi_lag_{lag}"]    = f["rsi_14"].shift(lag)

    # ── TARGET ────────────────────────────────────────────────────────────────
    future_return = c.pct_change(target_horizon).shift(-target_horizon)
    f["target"]         = (future_return > 0).astype(int)   # 1=sube, 0=baja
    f["future_return"]  = future_return                      # retorno real

    # Eliminar columnas originales que no son features
    drop_cols = ["open", "high", "low", "close", "volume", "quote_volume",
                 "trades", "sma_5", "sma_10"]
    f.drop(columns=[c for c in drop_cols if c in f.columns], inplace=True)

    return f.dropna()


if __name__ == "__main__":
    from data_fetcher import fetch_ohlcv
    print("Descargando datos...")
    df = fetch_ohlcv(interval="1h", limit=500)
    print(f"Datos: {df.shape}")
    
    features = build_features(df)
    print(f"\nFeatures generadas: {features.shape[1] - 2} features")
    print(f"Filas válidas:       {len(features)}")
    print(f"\nDistribución target:")
    print(features["target"].value_counts(normalize=True))
    print(f"\nPrimeras features:\n{list(features.columns[:15])}")
