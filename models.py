"""
models.py — ML Models profesionales para predicción de Bitcoin
Random Forest + XGBoost + Ensemble con walk-forward validation
Métricas: Accuracy, Precision, Recall, F1, Sharpe Ratio, Max Drawdown, Win Rate
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, classification_report, confusion_matrix)
from sklearn.model_selection import TimeSeriesSplit
import warnings
warnings.filterwarnings("ignore")

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("⚠️  XGBoost no instalado. Usando GradientBoosting como reemplazo.")
    print("   Instalar con: pip install xgboost")


# ─── Métricas de trading ──────────────────────────────────────────────────────

def sharpe_ratio(returns: pd.Series, risk_free: float = 0.0, periods: int = 8760) -> float:
    """Sharpe Ratio anualizado (periods=8760 para velas de 1h)."""
    excess = returns - risk_free / periods
    if excess.std() == 0:
        return 0.0
    return float(excess.mean() / excess.std() * np.sqrt(periods))


def max_drawdown(equity_curve: pd.Series) -> float:
    """Máximo Drawdown como porcentaje."""
    peak = equity_curve.cummax()
    drawdown = (equity_curve - peak) / peak
    return float(drawdown.min())


def calmar_ratio(returns: pd.Series, periods: int = 8760) -> float:
    """Retorno anual / Max Drawdown absoluto."""
    equity = (1 + returns).cumprod()
    mdd    = abs(max_drawdown(equity))
    ann_return = (1 + returns.mean()) ** periods - 1
    return ann_return / mdd if mdd != 0 else 0.0


def trading_metrics(predictions: np.ndarray, actual_returns: pd.Series) -> dict:
    """
    Calcula métricas de trading basadas en predicciones.
    Long si pred=1, Cash si pred=0.
    """
    pred_series  = pd.Series(predictions, index=actual_returns.index)
    strat_returns = actual_returns * pred_series   # solo operar cuando pred=1
    
    equity_curve = (1 + strat_returns).cumprod()
    bnh_equity   = (1 + actual_returns).cumprod()  # Buy & Hold

    win_trades  = (strat_returns[strat_returns != 0] > 0).sum()
    total_trades = (strat_returns != 0).sum()

    return {
        "sharpe":         sharpe_ratio(strat_returns),
        "max_drawdown":   max_drawdown(equity_curve),
        "calmar":         calmar_ratio(strat_returns),
        "win_rate":       win_trades / total_trades if total_trades > 0 else 0,
        "total_trades":   int(total_trades),
        "total_return":   float(equity_curve.iloc[-1] - 1),
        "bnh_return":     float(bnh_equity.iloc[-1] - 1),
        "avg_trade_return": float(strat_returns[strat_returns != 0].mean()),
        "profit_factor":  (strat_returns[strat_returns > 0].sum() /
                           abs(strat_returns[strat_returns < 0].sum())
                           if strat_returns[strat_returns < 0].sum() != 0 else np.inf),
    }


# ─── Modelos ──────────────────────────────────────────────────────────────────

def build_random_forest(n_estimators=300, max_depth=10, **kwargs):
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=20,
        max_features="sqrt",
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
        **kwargs
    )


def build_xgboost(**kwargs):
    if XGBOOST_AVAILABLE:
        return XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=10,
            scale_pos_weight=1,
            eval_metric="logloss",
            random_state=42,
            verbosity=0,
            **kwargs
        )
    else:
        return GradientBoostingClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            random_state=42,
            **kwargs
        )


# ─── Walk-Forward Validation ──────────────────────────────────────────────────

class WalkForwardValidator:
    """
    Validación correcta para series temporales: no hay data leakage.
    Entrena en pasado, predice en futuro progresivamente.
    """

    def __init__(self, n_splits: int = 5, test_size: int = 100):
        self.n_splits  = n_splits
        self.test_size = test_size
        self.results_  = []

    def validate(self, X: pd.DataFrame, y: pd.Series,
                 actual_returns: pd.Series, model_fn) -> dict:

        # Dynamically adjust test_size to fit available data
        n_samples = len(X)
        max_test_size = n_samples // (self.n_splits + 1)
        test_size = min(self.test_size, max_test_size)
        test_size = max(test_size, 10)  # minimum 10 samples per fold

        tscv    = TimeSeriesSplit(n_splits=self.n_splits, test_size=test_size)
        scaler  = StandardScaler()
        all_preds, all_true, all_returns = [], [], []

        print(f"🔄 Walk-Forward Validation ({self.n_splits} folds)...")

        for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
            X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
            y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]

            X_tr_s = scaler.fit_transform(X_tr)
            X_te_s = scaler.transform(X_te)

            model = model_fn()
            model.fit(X_tr_s, y_tr)
            preds = model.predict(X_te_s)

            fold_acc = accuracy_score(y_te, preds)
            self.results_.append({"fold": fold + 1, "accuracy": fold_acc, "samples": len(y_te)})
            print(f"   Fold {fold+1}: Accuracy={fold_acc:.3f}  ({len(y_te)} muestras)")

            all_preds.extend(preds)
            all_true.extend(y_te)
            all_returns.extend(actual_returns.iloc[test_idx])

        preds_arr   = np.array(all_preds)
        true_arr    = np.array(all_true)
        returns_ser = pd.Series(all_returns)

        metrics = {
            "accuracy":  accuracy_score(true_arr, preds_arr),
            "precision": precision_score(true_arr, preds_arr, zero_division=0),
            "recall":    recall_score(true_arr, preds_arr, zero_division=0),
            "f1":        f1_score(true_arr, preds_arr, zero_division=0),
        }
        metrics.update(trading_metrics(preds_arr, returns_ser))
        return metrics


# ─── Ensemble final ───────────────────────────────────────────────────────────

class BTCPredictor:
    """
    Predictor ensemble: Random Forest + XGBoost con voting por probabilidades.
    """

    def __init__(self, threshold: float = 0.55):
        self.threshold = threshold
        self.rf  = build_random_forest()
        self.xgb = build_xgboost()
        self.scaler = StandardScaler()
        self.feature_names_ = None
        self.is_fitted_ = False

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.feature_names_ = list(X.columns)
        X_s = self.scaler.fit_transform(X)
        self.rf.fit(X_s, y)
        self.xgb.fit(X_s, y)
        self.is_fitted_ = True
        print(f"✅ Modelos entrenados con {len(X)} muestras y {X.shape[1]} features")
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        X_s  = self.scaler.transform(X)
        p_rf = self.rf.predict_proba(X_s)[:, 1]
        p_xg = self.xgb.predict_proba(X_s)[:, 1]
        return (p_rf + p_xg) / 2   # promedio de probabilidades

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        proba = self.predict_proba(X)
        return (proba >= self.threshold).astype(int)

    def predict_next(self, X_last: pd.DataFrame) -> dict:
        """Predicción para la próxima vela."""
        proba = self.predict_proba(X_last)[0]
        pred  = int(proba >= self.threshold)
        return {
            "prediction":   "🟢 LONG (COMPRAR)" if pred == 1 else "🔴 CASH (ESPERAR)",
            "confidence":   float(proba),
            "signal":       pred,
            "threshold":    self.threshold,
            "bull_prob":    float(proba),
            "bear_prob":    float(1 - proba),
        }

    def feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        imp_rf  = self.rf.feature_importances_
        imp_xgb = self.xgb.feature_importances_
        avg_imp = (imp_rf + imp_xgb) / 2
        df = pd.DataFrame({
            "feature":    self.feature_names_,
            "importance": avg_imp,
            "rf_imp":     imp_rf,
            "xgb_imp":    imp_xgb,
        }).sort_values("importance", ascending=False)
        return df.head(top_n)


def full_training_pipeline(df_features: pd.DataFrame,
                            target_col: str = "target",
                            return_col: str = "future_return",
                            val_splits: int = 5) -> tuple:
    """
    Pipeline completo: validación → entrenamiento en todos los datos → predicción.
    
    Returns:
        (predictor_entrenado, metricas_dict)
    """
    feature_cols = [c for c in df_features.columns if c not in [target_col, return_col]]
    X       = df_features[feature_cols]
    y       = df_features[target_col]
    returns = df_features[return_col]

    print(f"\n📊 Dataset: {X.shape[0]} muestras × {X.shape[1]} features")
    print(f"   Target balance: {y.mean()*100:.1f}% bullish\n")

    # Walk-forward validation
    validator = WalkForwardValidator(n_splits=val_splits, test_size=100)
    val_metrics = validator.validate(X, y, returns, build_random_forest)

    print(f"\n{'='*50}")
    print(f"📈 MÉTRICAS DE VALIDACIÓN")
    print(f"{'='*50}")
    print(f"  Accuracy:       {val_metrics['accuracy']:.3f}")
    print(f"  Precision:      {val_metrics['precision']:.3f}")
    print(f"  Recall:         {val_metrics['recall']:.3f}")
    print(f"  F1 Score:       {val_metrics['f1']:.3f}")
    print(f"  Sharpe Ratio:   {val_metrics['sharpe']:.2f}")
    print(f"  Max Drawdown:   {val_metrics['max_drawdown']*100:.1f}%")
    print(f"  Win Rate:       {val_metrics['win_rate']*100:.1f}%")
    print(f"  Total Return:   {val_metrics['total_return']*100:.1f}%")
    print(f"  Buy&Hold:       {val_metrics['bnh_return']*100:.1f}%")
    print(f"  Profit Factor:  {val_metrics['profit_factor']:.2f}")
    print(f"  Total Trades:   {val_metrics['total_trades']}")
    print(f"{'='*50}\n")

    # Entrenar con todos los datos
    print("🏋️  Entrenando modelo final con todos los datos...")
    predictor = BTCPredictor(threshold=0.55)
    predictor.fit(X, y)

    return predictor, val_metrics


if __name__ == "__main__":
    from data_fetcher import fetch_ohlcv
    from features import build_features

    df   = fetch_ohlcv(interval="1h", limit=500)
    feat = build_features(df)

    predictor, metrics = full_training_pipeline(feat)

    print("\n🔮 PREDICCIÓN PRÓXIMA VELA:")
    feature_cols = [c for c in feat.columns if c not in ["target", "future_return"]]
    last_row     = feat[feature_cols].iloc[[-1]]
    signal       = predictor.predict_next(last_row)

    print(f"  Señal:      {signal['prediction']}")
    print(f"  Confianza:  {signal['confidence']*100:.1f}%")
    print(f"  Bull prob:  {signal['bull_prob']*100:.1f}%")
    print(f"  Bear prob:  {signal['bear_prob']*100:.1f}%")

    print("\n🏆 TOP 10 FEATURES MÁS IMPORTANTES:")
    print(predictor.feature_importance(10).to_string(index=False))
