"""
main.py — Bitcoin ML Predictor
Punto de entrada principal: conecta Binance → Features → ML → Dashboard
"""

import sys
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from data_fetcher  import fetch_ohlcv, fetch_current_price, fetch_order_book_depth
from features      import build_features
from models        import full_training_pipeline
from dashboard     import plot_main_dashboard, plot_metrics_summary, show_signal_banner

# ─── Configuración ────────────────────────────────────────────────────────────

CONFIG = {
    "symbol":         "BTCUSDT",
    "interval":       "1h",        # '15m', '1h', '4h', '1d'
    "candles":        500,          # historial a descargar
    "target_horizon": 1,            # velas adelante a predecir
    "val_splits":     5,            # folds de walk-forward validation
    "confidence_threshold": 0.55,   # mínima confianza para señal
    "save_plots":     True,
}


def run():
    print("=" * 60)
    print("  ₿  BITCOIN ML PREDICTOR — Powered by Binance API")
    print("=" * 60)
    print(f"\n⚙️  Configuración:")
    for k, v in CONFIG.items():
        print(f"   {k:25s}: {v}")

    # ── 1. Datos en tiempo real ───────────────────────────────────────────────
    print(f"\n📡 Conectando con Binance...")
    try:
        price_info = fetch_current_price(CONFIG["symbol"])
        ob_info    = fetch_order_book_depth(CONFIG["symbol"])
        df         = fetch_ohlcv(CONFIG["symbol"], CONFIG["interval"], CONFIG["candles"])
    except Exception as e:
        print(f"❌ Error al conectar con Binance: {e}")
        sys.exit(1)

    print(f"\n💰 {CONFIG['symbol']} — ${price_info['price']:,.2f}")
    print(f"   Cambio 24h:    {price_info['change_24h']:+.2f}%")
    print(f"   Volumen 24h:   {price_info['volume_24h']:,.0f} BTC")
    print(f"   Buy Pressure:  {ob_info['buy_pressure']*100:.1f}%")
    print(f"   Spread:        ${ob_info['spread']:.2f}")

    # ── 2. Feature Engineering ────────────────────────────────────────────────
    print(f"\n🔧 Generando features técnicas...")
    features = build_features(df, target_horizon=CONFIG["target_horizon"])
    feature_cols = [c for c in features.columns if c not in ["target", "future_return"]]
    print(f"   {len(feature_cols)} features generadas sobre {len(features)} velas")

    # ── 3. Entrenamiento + Validación ML ─────────────────────────────────────
    predictor, metrics = full_training_pipeline(
        features,
        val_splits=CONFIG["val_splits"]
    )

    # ── 4. Predicción próxima vela ────────────────────────────────────────────
    last_row = features[feature_cols].iloc[[-1]]
    signal   = predictor.predict_next(last_row)
    show_signal_banner(signal, price_info["price"])

    # ── 5. Predicciones históricas (para visualizar) ──────────────────────────
    X_all      = features[feature_cols]
    predictions = predictor.predict(X_all)

    # ── 6. Dashboard ──────────────────────────────────────────────────────────
    print("📊 Generando dashboard...")
    df_plot  = df.iloc[-len(features):]
    feat_viz = features.reset_index(drop=True)

    fig1 = plot_main_dashboard(
        df_plot, feat_viz, predictions,
        title=f"Bitcoin ML Predictor — {CONFIG['interval']} — ${price_info['price']:,.0f}"
    )

    feat_imp = predictor.feature_importance(15)
    fig2 = plot_metrics_summary(metrics, feat_imp)

    if CONFIG["save_plots"]:
        fig1.savefig("btc_dashboard.png", dpi=150, bbox_inches="tight",
                     facecolor="#0D1117")
        fig2.savefig("btc_metrics.png",   dpi=150, bbox_inches="tight",
                     facecolor="#0D1117")
        print("   💾 Gráficos guardados: btc_dashboard.png / btc_metrics.png")

    plt.show()

    # ── 7. Resumen final ──────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("  📈 RESUMEN FINAL")
    print("="*60)
    print(f"  Accuracy:      {metrics['accuracy']*100:.1f}%")
    print(f"  F1 Score:      {metrics['f1']:.3f}")
    print(f"  Sharpe Ratio:  {metrics['sharpe']:.2f}")
    print(f"  Win Rate:      {metrics['win_rate']*100:.1f}%")
    print(f"  Max Drawdown:  {metrics['max_drawdown']*100:.1f}%")
    print(f"  Profit Factor: {metrics['profit_factor']:.2f}")
    str_ret = metrics['total_return']*100
    bnh_ret = metrics['bnh_return']*100
    edge    = str_ret - bnh_ret
    print(f"  Estrategia:    {str_ret:+.1f}%  vs  B&H: {bnh_ret:+.1f}%  (edge: {edge:+.1f}%)")
    print("="*60)
    print(f"\n  🔮 SEÑAL ACTUAL: {signal['prediction']}")
    print(f"     Confianza: {signal['confidence']*100:.1f}%")
    print()


if __name__ == "__main__":
    run()
