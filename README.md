# strat_freqtrade

Stratégie Freqtrade pilotée par l'IA.

## Fichier principal
- `strategies/AIHybridStrategy.py`

## Principe
La stratégie repose sur un modèle de **régression logistique** entraîné en rolling:
- Les indicateurs techniques (EMA, RSI, retours, volatilité, volume relatif) servent de **features ML**.
- Les décisions d'entrée/sortie sont prises **uniquement** via la probabilité `ai_prob_up` prédite par le modèle.
- Seuils utilisés par défaut:
  - Entrée long si `ai_prob_up > 0.60` et en amélioration.
  - Sortie si `ai_prob_up < 0.45` ou chute rapide de confiance.

## Dépendances
- `scikit-learn` est requis pour produire des signaux.
- En l'absence de `scikit-learn`, la stratégie passe en **mode sécurité** et n'ouvre pas de position.

## Exemple d'utilisation
1. Copier la stratégie dans votre dossier utilisateur Freqtrade:
   - `user_data/strategies/AIHybridStrategy.py`
2. Installer les dépendances:
   - `pip install scikit-learn`
3. Lancer un backtest:
   - `freqtrade backtesting --strategy AIHybridStrategy --timeframe 5m`

## Notes
- Ajustez les seuils de confiance IA (`0.60`, `0.45`) selon vos paires et votre horizon.
- Validez toujours par backtest, puis dry-run avant passage en réel.
