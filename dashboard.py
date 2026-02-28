"""
dashboard.py — Dashboard profesional de visualización
Precio + indicadores + señales ML + métricas de backtest
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch
import matplotlib.dates as mdates
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

# Estilo oscuro profesional
plt.style.use("dark_background")
COLORS = {
    "bull":   "#00C805",
    "bear":   "#FF3B30",
    "price":  "#F7931A",   # Bitcoin orange
    "vol":    "#4A9EFF",
    "signal": "#FFD700",
    "bg":     "#0D1117",
    "panel":  "#161B22",
    "grid":   "#21262D",
    "text":   "#C9D1D9",
    "macd":   "#58A6FF",
    "rsi":    "#BC8CFF",
    "bb":     "#388BFD",
}


def plot_main_dashboard(df: pd.DataFrame, features: pd.DataFrame,
                        predictions: np.ndarray = None,
                        title: str = "Bitcoin ML Predictor — Dashboard") -> plt.Figure:
    """
    Dashboard principal de 5 paneles:
    1. Precio + Bollinger Bands + señales ML
    2. Volumen
    3. RSI
    4. MACD
    5. Equity curve vs Buy & Hold
    """

    fig = plt.figure(figsize=(18, 14), facecolor=COLORS["bg"])
    gs  = gridspec.GridSpec(5, 1, height_ratios=[4, 1.2, 1.2, 1.2, 1.8],
                             hspace=0.08, left=0.07, right=0.97,
                             top=0.94, bottom=0.05)

    # Índice de tiempo
    idx = df.index[-len(features):]
    close = df["close"].iloc[-len(features):]
    high  = df["high"].iloc[-len(features):]
    low   = df["low"].iloc[-len(features):]
    vol   = df["volume"].iloc[-len(features):]

    # ── Panel 1: Precio + BB + señales ────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0])
    ax1.set_facecolor(COLORS["panel"])
    ax1.plot(idx, close, color=COLORS["price"], linewidth=1.5, label="BTC/USDT", zorder=3)

    if "bb_upper" in features.columns:
        ax1.fill_between(idx, features["bb_upper"].values, features["bb_lower"].values,
                         alpha=0.1, color=COLORS["bb"], label="Bollinger Bands")
        ax1.plot(idx, features["bb_upper"].values, color=COLORS["bb"], alpha=0.4, linewidth=0.8)
        ax1.plot(idx, features["bb_lower"].values, color=COLORS["bb"], alpha=0.4, linewidth=0.8)
        ax1.plot(idx, features["bb_mid"].values,   color=COLORS["bb"], alpha=0.6,
                 linewidth=0.8, linestyle="--")

    if "ema_20" in features.columns:
        ax1.plot(idx, features["ema_20"].values, color="#FFA500", alpha=0.7,
                 linewidth=1.0, label="EMA 20")
    if "ema_50" in features.columns:
        ax1.plot(idx, features["ema_50"].values, color="#FF69B4", alpha=0.7,
                 linewidth=1.0, label="EMA 50")

    # Señales ML
    if predictions is not None and len(predictions) == len(idx):
        buy_mask  = predictions == 1
        sell_mask = predictions == 0
        ax1.scatter(idx[buy_mask],  close.values[buy_mask],
                    marker="^", color=COLORS["bull"], s=60, zorder=5,
                    label=f"Long ({buy_mask.sum()})", alpha=0.9)
        ax1.scatter(idx[sell_mask], close.values[sell_mask],
                    marker="v", color=COLORS["bear"], s=60, zorder=5,
                    label=f"Cash ({sell_mask.sum()})", alpha=0.5)

    ax1.set_title(title, color=COLORS["text"], fontsize=14, pad=10, fontweight="bold")
    ax1.legend(loc="upper left", fontsize=8, framealpha=0.3)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax1.tick_params(labelbottom=False)
    _style_ax(ax1)

    # ── Panel 2: Volumen ──────────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax2.set_facecolor(COLORS["panel"])
    colors_vol = [COLORS["bull"] if c >= o else COLORS["bear"]
                  for c, o in zip(df["close"].iloc[-len(features):],
                                   df["open"].iloc[-len(features):])]
    ax2.bar(idx, vol.values, color=colors_vol, alpha=0.7, width=0.03)
    if "volume_sma20" in features.columns:
        ax2.plot(idx, features["volume_sma20"].values,
                 color=COLORS["vol"], linewidth=1.0, alpha=0.8)
    ax2.set_ylabel("Volume", color=COLORS["text"], fontsize=8)
    ax2.tick_params(labelbottom=False)
    _style_ax(ax2)

    # ── Panel 3: RSI ──────────────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    ax3.set_facecolor(COLORS["panel"])
    if "rsi_14" in features.columns:
        rsi_vals = features["rsi_14"].values
        ax3.plot(idx, rsi_vals, color=COLORS["rsi"], linewidth=1.2, label="RSI 14")
        ax3.axhline(70, color=COLORS["bear"],  alpha=0.5, linewidth=0.8, linestyle="--")
        ax3.axhline(30, color=COLORS["bull"],  alpha=0.5, linewidth=0.8, linestyle="--")
        ax3.axhline(50, color=COLORS["text"],  alpha=0.2, linewidth=0.6)
        ax3.fill_between(idx, rsi_vals, 70,
                         where=(rsi_vals > 70), alpha=0.2, color=COLORS["bear"])
        ax3.fill_between(idx, rsi_vals, 30,
                         where=(rsi_vals < 30), alpha=0.2, color=COLORS["bull"])
    ax3.set_ylim(0, 100)
    ax3.set_ylabel("RSI", color=COLORS["text"], fontsize=8)
    ax3.tick_params(labelbottom=False)
    _style_ax(ax3)

    # ── Panel 4: MACD ─────────────────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[3], sharex=ax1)
    ax4.set_facecolor(COLORS["panel"])
    if "macd" in features.columns:
        ax4.plot(idx, features["macd"].values,     color=COLORS["macd"],  linewidth=1.0, label="MACD")
        ax4.plot(idx, features["macd_sig"].values, color=COLORS["signal"], linewidth=1.0, label="Signal")
        hist = features["macd_hist"].values
        bar_colors = [COLORS["bull"] if h >= 0 else COLORS["bear"] for h in hist]
        ax4.bar(idx, hist, color=bar_colors, alpha=0.6, width=0.03)
        ax4.axhline(0, color=COLORS["text"], alpha=0.3, linewidth=0.6)
    ax4.set_ylabel("MACD", color=COLORS["text"], fontsize=8)
    ax4.legend(loc="upper left", fontsize=7, framealpha=0.3)
    ax4.tick_params(labelbottom=False)
    _style_ax(ax4)

    # ── Panel 5: Equity curve ─────────────────────────────────────────────────
    ax5 = fig.add_subplot(gs[4], sharex=ax1)
    ax5.set_facecolor(COLORS["panel"])
    if predictions is not None:
        future_ret = features["future_return"].values
        pred_ret   = future_ret * predictions
        strategy   = pd.Series(pred_ret).replace(0, np.nan).fillna(0)
        equity_str = (1 + strategy).cumprod()
        equity_bnh = (1 + pd.Series(future_ret)).cumprod()

        ax5.plot(idx[:len(equity_str)], equity_str.values,
                 color=COLORS["bull"],  linewidth=1.5, label="ML Strategy")
        ax5.plot(idx[:len(equity_bnh)], equity_bnh.values,
                 color=COLORS["price"], linewidth=1.5, label="Buy & Hold", alpha=0.7)
        ax5.axhline(1, color=COLORS["text"], alpha=0.3, linewidth=0.6)
        ax5.fill_between(idx[:len(equity_str)], 1, equity_str.values,
                         where=(equity_str.values > 1), alpha=0.15, color=COLORS["bull"])
        ax5.fill_between(idx[:len(equity_str)], 1, equity_str.values,
                         where=(equity_str.values < 1), alpha=0.15, color=COLORS["bear"])

    ax5.set_ylabel("Equity", color=COLORS["text"], fontsize=8)
    ax5.legend(loc="upper left", fontsize=8, framealpha=0.3)
    ax5.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
    ax5.xaxis.set_major_locator(mdates.AutoDateLocator())
    _style_ax(ax5)

    plt.setp(ax5.xaxis.get_majorticklabels(), rotation=30, ha="right", fontsize=7)

    return fig


def plot_metrics_summary(metrics: dict, feature_importance: pd.DataFrame = None) -> plt.Figure:
    """Panel de métricas ML + importancia de features."""

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), facecolor=COLORS["bg"])

    # ── Métricas ──────────────────────────────────────────────────────────────
    ax = axes[0]
    ax.set_facecolor(COLORS["panel"])
    ax.set_title("📊 Métricas del Modelo", color=COLORS["text"], fontsize=12, pad=10)

    metric_labels = {
        "accuracy":    "Accuracy",
        "precision":   "Precision",
        "recall":      "Recall",
        "f1":          "F1 Score",
        "win_rate":    "Win Rate",
        "sharpe":      "Sharpe Ratio",
        "max_drawdown":"Max Drawdown",
        "total_return":"Total Return",
        "bnh_return":  "B&H Return",
        "profit_factor": "Profit Factor",
    }

    y_pos = list(range(len(metric_labels)))
    labels, values, bar_colors = [], [], []

    for key, label in metric_labels.items():
        val = metrics.get(key, 0)
        labels.append(label)
        values.append(abs(val) if key == "max_drawdown" else val)
        if key == "max_drawdown":
            bar_colors.append(COLORS["bear"])
        elif val >= 0:
            bar_colors.append(COLORS["bull"])
        else:
            bar_colors.append(COLORS["bear"])

    bars = ax.barh(y_pos, values, color=bar_colors, alpha=0.8, edgecolor="none")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, color=COLORS["text"], fontsize=9)
    ax.tick_params(colors=COLORS["text"])

    for i, (bar, val, key) in enumerate(zip(bars, values, metric_labels.keys())):
        orig_val = metrics.get(key, 0)
        if key in ["accuracy", "precision", "recall", "f1", "win_rate"]:
            label = f"{orig_val*100:.1f}%"
        elif key == "max_drawdown":
            label = f"{orig_val*100:.1f}%"
        elif key in ["total_return", "bnh_return"]:
            label = f"{orig_val*100:.1f}%"
        elif key == "profit_factor" and orig_val == np.inf:
            label = "∞"
        else:
            label = f"{orig_val:.2f}"
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                label, va="center", color=COLORS["text"], fontsize=8)

    _style_ax(ax)

    # ── Feature Importance ────────────────────────────────────────────────────
    ax2 = axes[1]
    ax2.set_facecolor(COLORS["panel"])
    ax2.set_title("🏆 Top Features Más Importantes", color=COLORS["text"], fontsize=12, pad=10)

    if feature_importance is not None:
        top = feature_importance.head(15)
        ax2.barh(range(len(top)), top["importance"].values,
                 color=COLORS["price"], alpha=0.8, edgecolor="none")
        ax2.set_yticks(range(len(top)))
        ax2.set_yticklabels(top["feature"].values, color=COLORS["text"], fontsize=8)
    
    _style_ax(ax2)

    fig.tight_layout(pad=2)
    return fig


def _style_ax(ax):
    ax.grid(True, color=COLORS["grid"], linewidth=0.5, alpha=0.5)
    ax.tick_params(colors=COLORS["text"], labelsize=7)
    for spine in ax.spines.values():
        spine.set_color(COLORS["grid"])


def show_signal_banner(signal: dict, current_price: float):
    """Imprime banner de la señal actual en consola."""
    color = "\033[92m" if signal["signal"] == 1 else "\033[91m"
    reset = "\033[0m"
    print(f"\n{'='*55}")
    print(f"  🔮 SEÑAL ML PARA PRÓXIMA VELA")
    print(f"{'='*55}")
    print(f"  💰 Precio actual:  ${current_price:,.2f}")
    print(f"  {color}📡 Señal:          {signal['prediction']}{reset}")
    print(f"  📊 Confianza:      {signal['confidence']*100:.1f}%")
    print(f"  🟢 Prob. Alcista:  {signal['bull_prob']*100:.1f}%")
    print(f"  🔴 Prob. Bajista:  {signal['bear_prob']*100:.1f}%")
    print(f"{'='*55}\n")
