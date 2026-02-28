"""
Stratégie Freqtrade pilotée par l'IA (régression logistique).

Objectif:
- Construire des features techniques servant uniquement d'entrées au modèle ML.
- Générer les signaux d'entrée/sortie à partir de la probabilité prédite par l'IA.

Note:
- `scikit-learn` est requis. Si indisponible, la stratégie se met en mode sécurité
  et n'ouvre aucune position.
"""

from __future__ import annotations

from typing import Any

import numpy as np
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
    """Stratégie orientée IA: les décisions sont prises par le score du modèle."""

    INTERFACE_VERSION = 3

    timeframe = "5m"
    can_short = False
    startup_candle_count = 300

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
        """Calcule les features et la probabilité haussière fournie par l'IA."""
        # Features techniques (utilisées comme variables explicatives du modèle)
        dataframe["ema_fast"] = dataframe["close"].ewm(span=12, adjust=False).mean()
        dataframe["ema_slow"] = dataframe["close"].ewm(span=26, adjust=False).mean()

        delta = dataframe["close"].diff()
        gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
        loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
        rs = gain / loss.replace(0, np.nan)
        dataframe["rsi"] = (100 - (100 / (1 + rs))).fillna(50)

        dataframe["ret_1"] = dataframe["close"].pct_change(1)
        dataframe["ret_3"] = dataframe["close"].pct_change(3)
        dataframe["ret_12"] = dataframe["close"].pct_change(12)
        dataframe["vol_24"] = dataframe["ret_1"].rolling(24).std()

        dataframe["vol_mean_24"] = dataframe["volume"].rolling(24).mean()
        dataframe["rel_volume"] = dataframe["volume"] / dataframe["vol_mean_24"].replace(0, np.nan)

        # Label supervisé: hausse > 0.3% d'ici 6 bougies
        future_return = dataframe["close"].shift(-6) / dataframe["close"] - 1
        dataframe["target_up"] = (future_return > 0.003).astype(int)

        # Mode sécurité: aucune entrée si sklearn indisponible
        dataframe["ai_prob_up"] = 0.0

        if SKLEARN_AVAILABLE and len(dataframe) > 240:
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

            if len(train_df) > 220:
                split_idx = int(len(train_df) * 0.8)
                x_train = train_df.iloc[:split_idx][feature_cols]
                y_train = train_df.iloc[:split_idx]["target_up"]
                x_pred = train_df.iloc[split_idx:][feature_cols]
                pred_index = train_df.iloc[split_idx:].index

                if y_train.nunique() > 1 and len(x_pred) > 0:
                    model = Pipeline(
                        [
                            ("scaler", StandardScaler()),
                            ("clf", LogisticRegression(max_iter=400, class_weight="balanced")),
                        ]
                    )
                    model.fit(x_train, y_train)
                    dataframe.loc[pred_index, "ai_prob_up"] = model.predict_proba(x_pred)[:, 1]

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict[str, Any]) -> DataFrame:
        """Entrée uniquement basée sur la confiance IA."""
        dataframe.loc[
            (
                (dataframe["ai_prob_up"] > 0.60)
                & (dataframe["ai_prob_up"].diff() > 0)
                & (dataframe["volume"] > 0)
            ),
            ["enter_long", "enter_tag"],
        ] = (1, "ai_confidence_long")

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict[str, Any]) -> DataFrame:
        """Sortie uniquement basée sur la perte de confiance IA."""
        dataframe.loc[
            (
                (
                    (dataframe["ai_prob_up"] < 0.45)
                    | (dataframe["ai_prob_up"].diff() < -0.08)
                )
                & (dataframe["volume"] > 0)
            ),
            ["exit_long", "exit_tag"],
        ] = (1, "ai_confidence_exit")

        return dataframe
