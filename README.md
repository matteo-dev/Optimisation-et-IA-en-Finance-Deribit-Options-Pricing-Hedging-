# 📈 Deribit Options Pricing, SSVI Calibration & Dynamic Hedging

Plateforme quantitative complète développée en Python et Streamlit pour l'analyse, le pricing et la couverture dynamique de produits dérivés sur les cryptomonnaies (BTC/ETH) à partir des données en temps réel de l'exchange Deribit.

## 🚀 Fonctionnalités Clés par Module

1. **Module 1 : Pipeline de Données**
   - Connexion à l'API publique de Deribit pour l'extraction des instruments et du carnet d'ordres.
   - Nettoyage des données, calcul du *mid-price*, filtrage de liquidité (spread max 25%) et de *moneyness*.
   - Test d'arbitrage statistique comparant la prime aux valeurs intrinsèques.

2. **Module 2 : Courbe des Taux (Nelson-Siegel)**
   - Extraction des taux d'intérêt implicites sans risque via la parité Call-Put.
   - Lissage et extrapolation de la structure par terme des taux à l'aide du modèle paramétrique de Nelson-Siegel.

3. **Module 3 : Surface de Volatilité (SSVI)**
   - Calcul de la volatilité implicite via Newton-Raphson et dichotomie, et extraction des Grecques.
   - Calibration en deux étapes du modèle SSVI (Gatheral & Jacquier) pour modéliser le *smile* de volatilité sans opportunité d'arbitrage.
   - Visualisation interactive de la surface SSVI en 3D.

4. **Module 4 : Structuration & Hedging**
   - Création de stratégies d'options complexes hors-grille (Call Spread, Put Spread, Straddle, Strangle) et calcul de leurs profils de payoff.
   - Optimisation d'un portefeuille de couverture auto-finançant **Delta-Gamma-Vega neutre** via l'optimiseur `scipy.optimize`.
   - Réalisation de *Stress Tests* pour évaluer la résistance du portefeuille face à des chocs de marché.

---

## 🛠️ Installation et Utilisation

1. **Cloner le dépôt :**
   ```bash
   git clone [https://github.com/votre-nom-d-utilisateur/deribit-options-optimization-ai.git](https://github.com/votre-nom-d-utilisateur/deribit-options-optimization-ai.git)
   cd deribit-options-optimization-ai
2. **Installer les dépendances :**
   ```bash
   pip install -r requirements.txt
3. **Lancer le frontend :**
   ```bash
   streamlit run dashboard.py
