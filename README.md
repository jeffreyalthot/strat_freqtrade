# strat_freqtrade

Stratégie Freqtrade avec filtre IA léger.

## Fichier principal
- `strategies/AIHybridStrategy.py`

## Principe
La stratégie combine:
- **Trend-following** via EMA 12/26.
- **Momentum/qualité de setup** via RSI, retours glissants et volume relatif.
- **Filtre IA** via une régression logistique (si `scikit-learn` est installé) pour estimer la probabilité d'une hausse à court terme.

Si `scikit-learn` n'est pas disponible, la stratégie reste exécutable en mode purement technique (probabilité IA neutre à `0.5`).

## Exemple d'utilisation
1. Copier la stratégie dans votre dossier utilisateur Freqtrade:
   - `user_data/strategies/AIHybridStrategy.py`
2. Vérifier les dépendances:
   - `pip install scikit-learn` (optionnel mais recommandé)
3. Lancer un backtest:
   - `freqtrade backtesting --strategy AIHybridStrategy --timeframe 5m`

## Notes
- Cette stratégie est un **template de départ**: adaptez les seuils (`ai_prob_up`, RSI, ROI, stoploss) selon votre marché.
- Toujours valider par backtest, puis dry-run avant tout passage en réel.
