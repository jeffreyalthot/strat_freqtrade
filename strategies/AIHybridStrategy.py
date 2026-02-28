"""
Stratégie Freqtrade combinant des signaux techniques classiques
avec une couche IA légère (régression logistique).

Objectif:
- Générer des signaux d'entrée/sortie à partir d'indicateurs TA.
- Utiliser un modèle ML entraîné en rolling pour filtrer les entrées.

Dépendances optionnelles:
- scikit-learn (recommandé pour activer la couche IA)

Si scikit-learn n'est pas disponible, la stratégie reste fonctionnelle
et applique uniquement les règles techniques.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from freqtrade.strategy import IStrategy
from pandas import DataFrame

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False


class AIHybridStrategy(IStrategy):
    """Stratégie Spot/Perp orientée swing court avec filtre IA."""

    INTERFACE_VERSION = 3

    # Paramètres de base
    timeframe = "5m"
    can_short = False
    startup_candle_count = 250

    minimal_roi = {
        "0": 0.05,
        "30": 0.03,
        "90": 0.015,
        "180": 0,
    }
    stoploss = -0.10
    trailing_stop = True
    trailing_stop_positive = 0.01
    trailing_stop_positive_offset = 0.025
    trailing_only_offset_is_reached = True

    process_only_new_candles = True
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False

    def populate_indicators(self, dataframe: DataFrame, metadata: dict[str, Any]) -> DataFrame:
        """Calcule indicateurs techniques + score IA."""
        # EMA trend
        dataframe["ema_fast"] = dataframe["close"].ewm(span=12, adjust=False).mean()
        dataframe["ema_slow"] = dataframe["close"].ewm(span=26, adjust=False).mean()

        # RSI (Wilder simplifié)
        delta = dataframe["close"].diff()
        gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
        loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
        rs = gain / loss.replace(0, np.nan)
        dataframe["rsi"] = 100 - (100 / (1 + rs))
        dataframe["rsi"] = dataframe["rsi"].fillna(50)

        # Volatilité et momentum
        dataframe["ret_1"] = dataframe["close"].pct_change(1)
        dataframe["ret_3"] = dataframe["close"].pct_change(3)
        dataframe["ret_12"] = dataframe["close"].pct_change(12)
        dataframe["vol_24"] = dataframe["ret_1"].rolling(24).std()

        # Volume relatif
        dataframe["vol_mean_24"] = dataframe["volume"].rolling(24).mean()
        dataframe["rel_volume"] = dataframe["volume"] / dataframe["vol_mean_24"].replace(0, np.nan)

        # Cible pour entraînement local:
        # 1 si le prix est plus haut dans 6 bougies, sinon 0
        future_return = dataframe["close"].shift(-6) / dataframe["close"] - 1
        dataframe["target_up"] = (future_return > 0.003).astype(int)

        # Score IA par défaut (neutre)
        dataframe["ai_prob_up"] = 0.5

        # Entraînement rolling léger du modèle IA
        if SKLEARN_AVAILABLE and len(dataframe) > 220:
            feature_cols = [
                "ema_fast",
                "ema_slow",
                "rsi",
                "ret_1",
                "ret_3",
                "ret_12",
                "vol_24",
                "rel_volume",
            ]

            train_df = dataframe.dropna(subset=feature_cols + ["target_up"]).copy()

            # Éviter le data leakage:
            # on entraîne sur passé et on prédit les dernières barres exploitables.
            if len(train_df) > 200:
                split_idx = int(len(train_df) * 0.8)
                x_train = train_df.iloc[:split_idx][feature_cols]
                y_train = train_df.iloc[:split_idx]["target_up"]
                x_pred = train_df.iloc[split_idx:][feature_cols]
                pred_index = train_df.iloc[split_idx:].index

                # Nécessite au moins 2 classes pour entraîner
                if y_train.nunique() > 1 and len(x_pred) > 0:
                    model = Pipeline(
                        [
                            ("scaler", StandardScaler()),
                            ("clf", LogisticRegression(max_iter=400, class_weight="balanced")),
                        ]
                    )
                    model.fit(x_train, y_train)
                    probs = model.predict_proba(x_pred)[:, 1]

                    dataframe.loc[pred_index, "ai_prob_up"] = probs

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict[str, Any]) -> DataFrame:
        """Entrée long si trend + momentum + score IA suffisant."""
        dataframe.loc[
            (
                (dataframe["ema_fast"] > dataframe["ema_slow"])
                & (dataframe["rsi"] > 48)
                & (dataframe["rsi"] < 72)
                & (dataframe["rel_volume"] > 0.9)
                & (dataframe["ai_prob_up"] > 0.56)
                & (dataframe["volume"] > 0)
            ),
            ["enter_long", "enter_tag"],
        ] = (1, "trend_ai_long")

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict[str, Any]) -> DataFrame:
        """Sortie si renversement trend ou perte de confiance IA."""
        dataframe.loc[
            (
                (
                    (dataframe["ema_fast"] < dataframe["ema_slow"])
                    | (dataframe["rsi"] > 78)
                    | (dataframe["ai_prob_up"] < 0.42)
                )
                & (dataframe["volume"] > 0)
            ),
            ["exit_long", "exit_tag"],
        ] = (1, "trend_ai_exit")

        return dataframe
