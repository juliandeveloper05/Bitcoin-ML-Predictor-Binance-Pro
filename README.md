<p align="center">
  <img src="https://img.shields.io/badge/Bitcoin-F7931A?style=for-the-badge&logo=bitcoin&logoColor=white" alt="Bitcoin"/>
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" alt="Scikit-Learn"/>
  <img src="https://img.shields.io/badge/XGBoost-006400?style=for-the-badge&logo=xgboost&logoColor=white" alt="XGBoost"/>
  <img src="https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white" alt="Pandas"/>
  <img src="https://img.shields.io/badge/Binance-FCD535?style=for-the-badge&logo=binance&logoColor=black" alt="Binance"/>
</p>

<h1 align="center">₿ Bitcoin ML Predictor — Binance Pro</h1>

<p align="center">
  <strong>Predicción de dirección de precio de Bitcoin usando Machine Learning con datos reales de Binance API</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/version-1.0.0-blue?style=flat-square" alt="Version"/>
  <img src="https://img.shields.io/badge/license-MIT-green?style=flat-square" alt="License"/>
  <img src="https://img.shields.io/badge/status-active-success?style=flat-square" alt="Status"/>
  <img src="https://img.shields.io/badge/API-Binance%20Public-FCD535?style=flat-square" alt="Binance API"/>
</p>

---

## 📋 Descripción

Sistema profesional de predicción de Bitcoin que combina **datos en tiempo real de Binance**, **60+ indicadores técnicos** y un **ensemble de Machine Learning (Random Forest + XGBoost)** para generar señales de trading automatizadas.

El sistema descarga datos OHLCV, construye features profesionales, entrena modelos con **walk-forward validation** (sin data leakage), y produce un dashboard oscuro profesional con métricas de backtesting.

### 🎯 Objetivo
Predicción de dirección de precio a **1h, 4h y 1d** con señales de trading automatizadas y métricas de rendimiento.

---

## 🏗️ Arquitectura del Sistema

```
Binance API ──→ Data Fetcher ──→ Feature Engineering ──→ ML Models ──→ Dashboard
   │                │                    │                    │              │
   │           OHLCV + Price        79 features         RF + XGBoost    5 Paneles
   │          + Order Book       (RSI, MACD, BB,      Walk-Forward    Precio + BB
   │                              ATR, OBV...)        Validation      Volumen, RSI
   │                                                                  MACD, Equity
   └── Endpoints Públicos (sin API key requerida)
```

---

## 🚀 Quick Start

### 1. Clonar el repositorio

```bash
git clone https://github.com/juliandeveloper05/bitcoin-simulator.git
cd bitcoin-simulator
```

### 2. Instalar dependencias

```bash
pip install -r requirements.txt
```

O manualmente:

```bash
pip install numpy pandas scikit-learn matplotlib requests xgboost
```

### 3. Ejecutar

```bash
python main.py
```

> ⚡ No requiere API key — usa endpoints públicos de Binance

---

## 📦 Estructura del Proyecto

| Archivo | Responsabilidad |
|---------|----------------|
| `main.py` | 🎯 Punto de entrada — Pipeline completo: datos → features → ML → señal → gráficos |
| `data_fetcher.py` | 📡 Binance API — OHLCV histórico + precio actual + order book depth |
| `features.py` | 🔧 Feature Engineering — 79 indicadores técnicos profesionales |
| `models.py` | 🤖 ML Models — Ensemble RF+XGBoost con walk-forward validation |
| `dashboard.py` | 📊 Dashboard oscuro profesional de 5 paneles |
| `requirements.txt` | 📋 Dependencias del proyecto |

---

## 🔧 Features Técnicas (79 indicadores)

### 📈 Momentum
- **RSI** (7, 14, 21 períodos) + divergencia RSI
- **MACD** (línea, señal, histograma)
- **Stochastic** (%K, %D)
- **CCI** (Commodity Channel Index)
- **Williams %R**

### 📉 Volatilidad
- **Bollinger Bands** (superior, media, inferior, ancho, %B)
- **ATR** (Average True Range) normalizado
- **Volatilidad** anualizada (5 y 20 períodos)

### 💹 Precio
- **EMA** (5, 10, 20, 50, 100, 200)
- **SMA** (20, 50, 100, 200)
- **Retornos** en múltiples ventanas
- **Distancia a EMAs** (% de desviación)

### 📊 Volumen
- **OBV** (On-Balance Volume) + señal EMA
- **VWAP** (Volume Weighted Average Price)
- **Volume Ratio** (volumen relativo a SMA 20)
- **Volume Momentum** (cambio de volumen a 5 períodos)

### 🕯️ Patrones de Velas
- Body, Upper/Lower Shadow, Candle Direction, Doji, High-Low %

### 🔄 Tendencia
- Trend 20/50, Trend 50/200, Golden Cross, Death Cross

### 📐 Soporte/Resistencia
- Rolling High/Low 20, Distance to High/Low

### ⏳ Lag Features
- Return, Volume Ratio y RSI con lags de 1, 2, 3, 5, 8 períodos

---

## 🤖 Modelos de Machine Learning

### Ensemble: Random Forest + XGBoost

| Componente | Configuración |
|-----------|---------------|
| **Random Forest** | 300 árboles, max_depth=10, min_samples_leaf=20, balanced weights |
| **XGBoost** | 300 estimators, max_depth=6, lr=0.05, subsample=0.8 |
| **Ensemble** | Promedio de probabilidades con threshold adaptativo (0.55) |
| **Scaler** | StandardScaler para normalización de features |

### Walk-Forward Validation
- **Sin data leakage**: entrena en pasado, predice en futuro
- **TimeSeriesSplit** con tamaño de test dinámico
- **Métricas por fold**: Accuracy por cada ventana temporal

---

## 📊 Métricas de Backtesting

| Métrica | Descripción |
|---------|-------------|
| **Accuracy** | Porcentaje de predicciones correctas |
| **Precision / Recall / F1** | Métricas de clasificación |
| **Sharpe Ratio** | Retorno ajustado por riesgo (anualizado) |
| **Max Drawdown** | Máxima caída desde pico de equity |
| **Win Rate** | Porcentaje de trades ganadores |
| **Profit Factor** | Ganancias brutas / Pérdidas brutas |
| **Total Return** | Retorno total de la estrategia |
| **Buy & Hold Return** | Benchmark: mantener BTC |
| **Calmar Ratio** | Retorno anual / Max Drawdown |

---

## 📊 Dashboard

El sistema genera automáticamente **2 dashboards profesionales** con tema oscuro:

### Dashboard Principal (5 paneles)
1. **Precio + Bollinger Bands + señales ML** (Long ▲ / Cash ▼)
2. **Volumen** con SMA 20 (verde=alcista, rojo=bajista)
3. **RSI 14** con zonas de sobrecompra/sobreventa (70/30)
4. **MACD** (línea + señal + histograma)
5. **Equity Curve** — ML Strategy vs Buy & Hold

### Dashboard de Métricas
- **Barras horizontales** con todas las métricas del modelo
- **Top 15 Features** más importantes (importancia promedio RF+XGBoost)

---

## ⚙️ Configuración

Editar el diccionario `CONFIG` en `main.py`:

```python
CONFIG = {
    "symbol":         "BTCUSDT",     # Par de trading
    "interval":       "1h",          # '15m', '1h', '4h', '1d'
    "candles":        500,           # Historial (máx 1000)
    "target_horizon": 1,             # Velas adelante a predecir
    "val_splits":     5,             # Folds de validación
    "confidence_threshold": 0.55,    # Mínima confianza para señal
    "save_plots":     True,          # Guardar PNGs
}
```

---

## 🛠️ Tech Stack

| Tecnología | Uso |
|-----------|-----|
| **Python 3.11+** | Lenguaje principal |
| **NumPy** | Cálculos numéricos |
| **Pandas** | Manipulación de datos y series temporales |
| **scikit-learn** | Random Forest, StandardScaler, TimeSeriesSplit |
| **XGBoost** | Gradient Boosting optimizado |
| **Matplotlib** | Dashboards y visualización |
| **Requests** | Conexión con Binance API |
| **Binance API** | Datos OHLCV en tiempo real (endpoints públicos) |

---

## 📄 Output del Sistema

```
============================================================
  ₿  BITCOIN ML PREDICTOR — Powered by Binance API
============================================================

💰 BTCUSDT — $65,969.48
   Cambio 24h:    -1.97%
   Volumen 24h:   20,316 BTC
   Buy Pressure:  24.5%

🔧 79 features generadas sobre 299 velas

🔄 Walk-Forward Validation (5 folds)...
   Fold 1: Accuracy=0.490
   Fold 2: Accuracy=0.510
   ...

=======================================================
  🔮 SEÑAL ML PARA PRÓXIMA VELA
=======================================================
  💰 Precio actual:  $65,969.48
  📡 Señal:          🟢 LONG (COMPRAR)
  📊 Confianza:      58.2%
  🟢 Prob. Alcista:  58.2%
  🔴 Prob. Bajista:  41.8%
=======================================================

   💾 Gráficos guardados: btc_dashboard.png / btc_metrics.png
```

---

## ⚠️ Disclaimer

> Este proyecto es **exclusivamente educativo y de investigación**. No constituye asesoramiento financiero. El trading de criptomonedas conlleva riesgos significativos. Los resultados pasados no garantizan resultados futuros. **Úsalo bajo tu propia responsabilidad.**

---

## 👨‍💻 Author

**Julian Javier Soto**
Senior Software Engineer · AI & Audio Processing
Specialized in Python, TypeScript, React, Machine Learning & Cloud Deployment

[![GitHub](https://img.shields.io/badge/GitHub-juliandeveloper05-181717?style=for-the-badge&logo=github)](https://github.com/juliandeveloper05)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Julian%20Soto-0A66C2?style=for-the-badge&logo=linkedin)](https://www.linkedin.com/in/julian-soto)
[![Portfolio](https://img.shields.io/badge/Portfolio-juliansoto-000000?style=for-the-badge&logo=vercel)](https://juliansoto.dev)
[![Instagram](https://img.shields.io/badge/Instagram-paleo__0k21-E4405F?style=for-the-badge&logo=instagram&logoColor=white)](https://instagram.com/paleo_0k21)

📧 Email: juliansoto.dev@gmail.com
📱 WhatsApp: +54 9 11 3066-6369

---

## 📝 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

<p align="center">
  <strong>Bitcoin ML Predictor v1.0.0</strong> — Made with ❤️ and 🧠 by Julian Javier Soto · © 2026
</p>
