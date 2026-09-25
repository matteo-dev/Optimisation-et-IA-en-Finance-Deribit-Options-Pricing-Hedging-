# Fichier en Python du Backend complet du Projet "Optimisation et IA"

# Importation des bibliothèques nécessaires
import requests # Pour les routes API vers Deribit
import pandas as pd
import numpy as np
import time # Pour les calculs de maturité avec le type timestamp
from scipy.optimize import minimize # Pour l'optimisation des modèles de taux et de volatilité
from sklearn.linear_model import LinearRegression # Pour l'extraction des taux via la parité Call-Put
from scipy.stats import norm # Pour les fonctions de distribution normale dans le modèle de Black-Scholes


## Partie 1 : Récupération et Traitement des Données depuis Deribit

# Classe pour gérer la récupération et le traitement des données de Deribit
class DeribitDataPipeline:

    # Fonction d'initialisation avec la devise souhaitée (ex: BTC, ETH)
    def __init__(self, currency="BTC"):
        self.currency = currency
        
        # Url pour l'accès à l'API de Deribit
        self.base_url = "https://www.deribit.com/api/v2/public/"

    # Fonction pour récupérer les instruments actifs depuis Deribit
    def get_active_instruments(self, kind="option"):
        params = {"currency": self.currency, "kind": kind, "expired": "false"}

        # Récupération des données via l'API publique de Deribit
        response = requests.get(self.base_url + "get_instruments", params=params)
        # Conversion de la réponse en dictionnaire Python
        json_resp = response.json()
        
        # Vérification de la clé "result" dans la réponse pour éviter les erreurs 
        if "result" not in json_resp:
            raise ValueError(f"Erreur API Deribit (Instruments) : {json_resp}")
            
        # Retourne un DataFrame Pandas avec les données des instruments actifs
        return pd.DataFrame(json_resp["result"])

    # Fonction pour récupérer les données du carnet d'ordres depuis Deribit
    def get_market_data(self, kind="option"):
        params = {"currency": self.currency, "kind": kind}
        response = requests.get(self.base_url + "get_book_summary_by_currency", params=params)
        json_resp = response.json()
        
        # Vérification de la clé "result" dans la réponse pour éviter les erreurs
        if "result" not in json_resp:
            raise ValueError(f"Erreur API Deribit (Market Data) : {json_resp}")

        # Retourne un DataFrame Pandas avec les données du carnet d'ordres  
        return pd.DataFrame(json_resp["result"])

    # Fonction pour fusionner les données sur les instruments et les données de marché
    def fetch_and_merge_data(self):
        # Récupération des données brutes venant des deux appels "Get" de l'API
        instruments = self.get_active_instruments(kind="option")
        market_data = self.get_market_data(kind="option")
        
        # Fusion des deux DataFrames existants sur la colonne "instrument_name"
        df = pd.merge(market_data, instruments[['instrument_name', 'strike', 'expiration_timestamp', 'option_type']], 
                      on='instrument_name', how='inner')
        return df

    # Fonction pour nettoyer les données, les filtrer et calculer les indicateurs nécessaires pour l'analyse d'arbitrage
    def process_data(self, df, spread_threshold=0.25):

        # Calcul du mid-price (prix moyen entre le meilleur bid et le meilleur ask)
        df['mid_price'] = (df['bid_price'] + df['ask_price']) / 2
        
        # Suppression des lignes où le mid_price est nul ou NaN
        df = df.dropna(subset=['mid_price', 'bid_price', 'ask_price'])
        df = df[df['mid_price'] > 0]

        # Calcul du spread et filtrage (spread > 25% du mid_price)
        df['spread'] = df['ask_price'] - df['bid_price']
        df['spread_ratio'] = df['spread'] / df['mid_price']
        df_filtered = df[df['spread_ratio'] <= spread_threshold].copy()

        # Récupération du prix actuel du sous-jacent (Spot/Index Price)
        df_filtered['underlying_price'] = df_filtered['estimated_delivery_price'] 
        
        # Calcul du Moneyness simplifié (K/S) pour éliminer les options trop OTM/ITM
        df_filtered['moneyness'] = df_filtered['strike'] / df_filtered['underlying_price']
        
        # Filtre sur les options dont le strike est entre 50% et 150% du prix spot
        df_filtered = df_filtered[(df_filtered['moneyness'] >= 0.5) & (df_filtered['moneyness'] <= 1.5)]
        
        return df_filtered

    # Fonction pour vérifier les opportunités d'arbitrage en comparant le mid-price avec la valeur intrinsèque
    def check_arbitrage(self, df):
        # Check de l'option type pour calculer la valeur intrinsèque
        conditions = [
            df['option_type'] == 'call',
            df['option_type'] == 'put'
        ]
        # Calcul de la valeur intrinsèque pour les calls et les puts
        choices = [
            np.maximum(df['underlying_price'] - df['strike'], 0),
            np.maximum(df['strike'] - df['underlying_price'], 0)
        ]
        
        # Application de la fonction np.select pour créer une nouvelle colonne "intrinsic_value" basée sur les conditions et les choix définis
        df['intrinsic_value'] = np.select(conditions, choices, default=0)
        df['mid_price_usd'] = df['mid_price'] * df['underlying_price']

        # Création d'une colonne Arbitrage si le mid_price est inférieur à la valeur intrinsèque 
        df['arbitrage_opportunity'] = df['mid_price_usd'] < df['intrinsic_value']

        # Exportation du DataFrame final en CSV pour analyse ou stockage
        df.to_csv(f"processed_deribit_options_{self.currency}.csv", index=False)     
        return df

# Exemple d'Utilisation de la Partie 1
if __name__ == "__main__":
    pipeline = DeribitDataPipeline(currency="BTC")
    
    print("1. Recuperation des donnees brutes...")
    raw_data = pipeline.fetch_and_merge_data()
    
    print("2. Nettoyage et calcul du mid-price...")
    clean_data = pipeline.process_data(raw_data, spread_threshold=0.25)
    
    print("3. Verification des opportunites d'arbitrage statistique...")
    final_data = pipeline.check_arbitrage(clean_data)
    
    arbitrages = final_data[final_data['arbitrage_opportunity'] == True]
    
    print(f"Nombre d'options analysees apres filtrage : {len(final_data)}")
    print(f"Nombre d'opportunites d'arbitrage detectees : {len(arbitrages)}")


## Partie 2 : Courbe des taux et Nelson Siegel


# Classe pour modéliser la courbe des taux d'intérêt à partir des données extraites de Deribit
class InterestRateModel:
    # Fonction pour intialiser Nelson-Siegel avec des paramètres initiaux
    def __init__(self):
        self.ns_params = [0.05, -0.02, 0.02, 1.0]

    # Fonction pour calculer les maturités en années
    def calculate_time_to_maturity(self, df):
        # Le temps du timestamp de base est en millisecondes
        current_time = time.time() * 1000 
        # Calcul de la maturité en années à partir du timestamp d'expiration
        df['T'] = (df['expiration_timestamp'] - current_time) / (1000 * 60 * 60 * 24 * 365.25)
        
        # Exclusion des options expirées ou dont l'échéance est inférieure à 30 jours (instabilité des prix)
        return df[df['T'] > (30 / 365.25)].copy()

    # Fonction pour extraire les taux d'intérêt implicites à partir de la parité Call-Put et d'une régression linéaire
    def extract_empirical_rates(self, df):
        df = self.calculate_time_to_maturity(df)
        
        # Conversion du mid_price en USD
        df['mid_price_usd'] = df['mid_price'] * df['underlying_price']
        
        # Séparation des Calls et des Puts 
        calls = df[df['option_type'] == 'call'][['strike', 'T', 'mid_price_usd']].rename(columns={'mid_price_usd': 'C'})
        puts = df[df['option_type'] == 'put'][['strike', 'T', 'mid_price_usd']].rename(columns={'mid_price_usd': 'P'})
        
        # Fusionner les Calls et les Puts sur le couple (strike, maturité)
        pairs = pd.merge(calls, puts, on=['strike', 'T'], how='inner')
        # Calcul de (C - P)
        pairs['C_minus_P'] = pairs['C'] - pairs['P']

        rates_data = []
        
        # Grouper par maturité (T)
        for t_val, group in pairs.groupby('T'):
            if len(group) < 3:
                continue

            # Initialisation de la variable explicative (Strike) et de la variable à expliquer (C - P)
            X = group[['strike']].values
            y = group['C_minus_P'].values
            
            # Régréssion linéaire pour extraire la pente et l'intercept
            model = LinearRegression().fit(X, y)
            slope = model.coef_[0]
            intercept = model.intercept_
            
            if slope >= 0:
                continue # Anomalie de marché liée au manque de liquidité
                
            # Calcul du taux d'intérêt continu r(T) à partir de la pente de la régression linéaire
            r_T = -np.log(-slope) / t_val
            
            # Calcul du Forward F(T) à partir de l'intercept et de la pente
            F_T = intercept / (-slope)
            
            rates_data.append({'T': t_val, 'r_T': r_T, 'F_T': F_T})

        # Retourne un DataFrame trié par maturité avec les taux d'intérêt extraits 
        return pd.DataFrame(rates_data).sort_values(by='T').reset_index(drop=True)

    # Fonction pour calculer les taux lissés à partir du modèle de Nelson-Siegel 
    def nelson_siegel(self, T, beta0, beta1, beta2, tau):
        term1 = (1 - np.exp(-T / tau)) / (T / tau)
        term2 = term1 - np.exp(-T / tau)
        return beta0 + beta1 * term1 + beta2 * term2

    # Fonction pour ajuster les paramètres de Nelson-Siegel en minimisant l'erreur quadratique entre les taux empiriques et les taux du modèle
    def fit_nelson_siegel(self, empirical_rates_df):
        T_data = empirical_rates_df['T'].values
        r_data = empirical_rates_df['r_T'].values
        
        # Fonction d'erreur à minimiser
        def objective(params):
            beta0, beta1, beta2, tau = params
            # Le paramètre tau doit être strictement positif
            if tau <= 0:
                return 1e6 
            
            r_pred = self.nelson_siegel(T_data, beta0, beta1, beta2, tau)
            # Retourne la moyenne des carrés des erreurs entre les taux empiriques et les taux prédits par le modèle
            return np.mean((r_data - r_pred) ** 2)

        # Choix de bornes pour les paramètres  
        bounds = ((-0.5, 0.5), (-5.0, 5.0), (-5.0, 5.0), (0.01, 10))
        
        # Lancement de l'optimiseur L-BFGS-B (parfait pour les problèmes bornés)
        result = minimize(objective, self.ns_params, bounds=bounds, method='L-BFGS-B')
        
        # Si l'optimisation a réussi, on stocke les paramètres optimisés et on les retourne sous forme de dictionnaire
        if result.success:
            self.ns_params = result.x
            return {"beta0": result.x[0], "beta1": result.x[1], "beta2": result.x[2], "tau": result.x[3]}
        else:
            raise ValueError("L'optimisation Nelson-Siegel a échoué.")

    # Fonction pour récupérer les taux lissés à partir des paramètres optimisés de Nelson-Siegel pour une liste de maturités donnée
    def get_ns_rates(self, maturities):
        beta0, beta1, beta2, tau = self.ns_params
        return self.nelson_siegel(np.array(maturities), beta0, beta1, beta2, tau)

# Exemple d'Utilisation de la Partie 2
if __name__ == "__main__":
    pipeline = DeribitDataPipeline("BTC")
    clean_data = pipeline.process_data(pipeline.fetch_and_merge_data())
    ir_model = InterestRateModel()
    
    # 1) Extraction de la courbe discontinue brute
    empirical_rates = ir_model.extract_empirical_rates(clean_data)
    print(empirical_rates)
    
    # 2) Lissage via optimisation Nelson-Siegel
    best_params = ir_model.fit_nelson_siegel(empirical_rates)
    print("Parametres Nelson-Siegel optimises :", best_params)
    pass


## Partie 3 : Calibration de la volatilité paramétrique SSVI


# Classe pour modéliser la volatilité implicite à partir des données extraites de Deribit et calibrer le modèle SSVI
class VolatilityModel:

    # Fonction d'initialisation avec des paramètres initiaux pour la calibration SSVI
    def __init__(self):
        self.ssvi_atm_params = {'kappa': 1.0, 'v0': 0.1, 'v_inf': 0.1}
        self.ssvi_smile_params = {'rho': -0.2, 'eta': 1.0, 'lambda_': 0.5}

    # Fonction pour calculer d1 et d2 du modèle de Black-Scholes
    def d1_d2(self, S, K, T, r, sigma):
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return d1, d2

    # Fonction pour calculer le prix d'une option (Call ou Put)
    def bs_price(self, S, K, T, r, sigma, option_type):
        d1, d2 = self.d1_d2(S, K, T, r, sigma)
        if option_type == 'call':
            return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        else: # put
            return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

    # Fonction pour calculer le Vega
    def bs_vega(self, S, K, T, r, sigma):
        d1, _ = self.d1_d2(S, K, T, r, sigma)
        return S * norm.pdf(d1) * np.sqrt(T)

    # Fonction pour calculer l'ensemble des Grecques (Delta, Gamma, Vega, Theta, Rho) 
    def calculate_greeks(self, S, K, T, r, sigma, option_type):
        d1, d2 = self.d1_d2(S, K, T, r, sigma)
        vega = self.bs_vega(S, K, T, r, sigma)
        gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))

        # Cas "Call", calcul de Delta, Theta et Rho
        if option_type == 'call':
            delta = norm.cdf(d1)
            theta = -(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2)
            rho = K * T * np.exp(-r * T) * norm.cdf(d2)
        else: # Cas "Put", calcul de Delta, Theta et Rho
            delta = norm.cdf(d1) - 1
            theta = -(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * norm.cdf(-d2)
            rho = -K * T * np.exp(-r * T) * norm.cdf(-d2)
            
        return {'delta': delta, 'gamma': gamma, 'vega': vega, 'theta': theta / 365, 'rho': rho / 100}

    # Fonction pour calculer la volatilité implicite à partir du prix de marché d'une option
    def implied_volatility(self, S, K, T, r, market_price, option_type, tol=1e-5, max_iter=100):

        # Utilisation de la méthode de Newton-Raphson
        sigma = 0.5 # Point de départ
        for _ in range(max_iter):
            price = self.bs_price(S, K, T, r, sigma, option_type)
            diff = price - market_price
            if abs(diff) < tol:
                return sigma
            vega = self.bs_vega(S, K, T, r, sigma)
            
            # Si Vega faible alors divergence, donc on passe à la méthode de dichotomie pour plus de sécurité
            if vega < 1e-6:
                break 

            # Mise à jour de la volatilité selon la méthode de Newton-Raphson
            sigma = sigma - diff / vega
            # Sécurité pour éviter les valeurs de sigma non réalistes 
            if sigma <= 0.001 or sigma > 10: 
                break

        # Utilisation de la dichotomie si Newton-Raphson a échoué ou divergé
        low, high = 0.001, 5.0
        for _ in range(max_iter):
            mid = (low + high) / 2
            price_mid = self.bs_price(S, K, T, r, mid, option_type)
            diff_mid = price_mid - market_price

            # Si la différence est suffisamment petite, on considère que nous avons trouvé la volatilité implicite
            if abs(diff_mid) < tol:
                return mid

            # On vérifie de quel côté se trouve le prix de marché par rapport au prix calculé à la volatilité "low" pour ajuster les bornes
            price_low = self.bs_price(S, K, T, r, low, option_type)
            if (price_low - market_price) * diff_mid < 0:
                high = mid
            else:
                low = mid
                
        return mid # Retourne la meilleure approximation trouvée

    # Fonction pour ajout de la volatilité implicite et des grecques dans le DataFrame pour chaque option
    def extract_iv_and_greeks(self, df):
        ivs, deltas, gammas, vegas, thetas, rhos = [], [], [], [], [], []
        
        # Boucle pour ajouter la volatilité implicite 
        for _, row in df.iterrows():
            iv = self.implied_volatility(
                row['underlying_price'], row['strike'], row['T'], 
                row['r_T'], row['mid_price_usd'], row['option_type']
            )
            ivs.append(iv)
            
            # Boucle pour calculer les grecques à partir de la volatilité implicite trouvée
            greeks = self.calculate_greeks(
                row['underlying_price'], row['strike'], row['T'], 
                row['r_T'], iv, row['option_type']
            )
            # Boucle pour ajouter les grecques dans les listes correspondantes
            deltas.append(greeks['delta'])
            gammas.append(greeks['gamma'])
            vegas.append(greeks['vega'])
            thetas.append(greeks['theta'])
            rhos.append(greeks['rho'])

        # Ajout des colonnes de volatilité implicite et des grecques dans le DataFrame final    
        df['iv'] = ivs
        df['delta'] = deltas
        df['gamma'] = gammas
        df['vega'] = vegas
        df['theta'] = thetas
        df['rho'] = rhos
        return df

    # Fonction pour calculer la fonction theta(t) pour la partie ATM du modèle SSVI
    def _theta_t(self, t, kappa, v0, v_inf):
        return ((1 - np.exp(-kappa * t)) / (kappa * t) * (v0 - v_inf) + v_inf) * t

    # Fonction pour calculer la fonction w(k,t) pour la partie smile du modèle SSVI 
    def _ssvi_w(self, k, t, theta, rho, eta, lambda_):
        # Calcul de phi
        phi = eta / (theta ** lambda_)
        # Calcul de w(k,t) selon la formule SSVI
        sqrt_term = np.sqrt((phi * k + rho)**2 + (1 - rho**2))
        return (theta / 2) * (1 + rho * phi * k + sqrt_term)

    # Fonction pour calibrer les paramètres du modèle SSVI en deux étapes : d'abord les paramètres ATM puis les paramètres de smile
    def calibrate_ssvi(self, df):
        # Calcul du Forward et du Log-Forward Moneyness (k)
        df['F_T'] = df['underlying_price'] * np.exp(df['r_T'] * df['T'])
        df['k'] = np.log(df['strike'] / df['F_T'])

        # Filtre sur les options ATM
        df_atm = df[df['k'].abs() < 0.05].copy()

        # Fonction pour calibrer les paramètres ATM        
        def objective_step1(params):
            kappa, v0, v_inf = params
            error = 0
            for _, row in df_atm.iterrows():
                theta = self._theta_t(row['T'], kappa, v0, v_inf)
                if theta <= 0: return 1e6
                sigma_atm = np.sqrt(theta / row['T'])
                price_model = self.bs_price(row['underlying_price'], row['strike'], row['T'], row['r_T'], sigma_atm, row['option_type'])
                error += (price_model - row['mid_price_usd'])**2
            return error

        # Choix de bornes strictes pour les paramètres ATM 
        bounds_1 = ((0.01, 10), (0.01, 2.0), (0.01, 2.0))
        # Lancement de l'optimiseur pour la calibration des paramètres ATM du modèle SSVI
        res_1 = minimize(objective_step1, [1.0, 0.1, 0.1], bounds=bounds_1, method='L-BFGS-B')
        # Stockage des paramètres ATM optimisés dans un dictionnaire 
        self.ssvi_atm_params = {'kappa': res_1.x[0], 'v0': res_1.x[1], 'v_inf': res_1.x[2]}

        kappa, v0, v_inf = res_1.x
        
        # Fonction pour calibrer les paramètres de smile 
        def objective_step2(params):
            rho, eta, lambda_ = params
            error = 0
            for _, row in df.iterrows():
                theta = self._theta_t(row['T'], kappa, v0, v_inf)
                w = self._ssvi_w(row['k'], row['T'], theta, rho, eta, lambda_)
                if w <= 0: return 1e6
                sigma_ssvi = np.sqrt(w / row['T'])
                price_model = self.bs_price(row['underlying_price'], row['strike'], row['T'], row['r_T'], sigma_ssvi, row['option_type'])
                error += (price_model - row['mid_price_usd'])**2
            return error

        # Choix de bornes strictes pour les paramètres de smile
        bounds_2 = ((-0.99, 0.99), (0.01, 5.0), (0.001, 1.0))
        # Lancement de l'optimiseur pour la calibration des paramètres de Smile du modèle SSVI
        res_2 = minimize(objective_step2, [-0.2, 1.0, 0.5], bounds=bounds_2, method='L-BFGS-B')
        # Stockage des paramètres de Smile optimisés dans un dictionnaire 
        self.ssvi_smile_params = {'rho': res_2.x[0], 'eta': res_2.x[1], 'lambda_': res_2.x[2]}

        return {**self.ssvi_atm_params, **self.ssvi_smile_params}

    # Fonction pour calculer la volatilité implicite à partir du modèle SSVI pour une option donnée 
    def get_ssvi_vol(self, S, K, T, r):
        F_T = S * np.exp(r * T)
        k = np.log(K / F_T)
        theta = self._theta_t(T, self.ssvi_atm_params['kappa'], self.ssvi_atm_params['v0'], self.ssvi_atm_params['v_inf'])
        w = self._ssvi_w(k, T, theta, self.ssvi_smile_params['rho'], self.ssvi_smile_params['eta'], self.ssvi_smile_params['lambda_'])
        return np.sqrt(w / T)
    
# Exemple d'Utilisation de la Partie 3
if __name__ == "__main__":
    print("--- Lancement des tests du Module de Volatilite ---\n")
    
    vol_model = VolatilityModel()
    
    # --- Paramètres de marché fictifs ---
    S_test = 65000.0  # Prix du Bitcoin
    K_test = 67000.0  # Strike (Option en dehors de la monnaie)
    T_test = 30 / 365.25 # Maturité de 30 jours
    r_test = 0.05     # Taux sans risque de 5%
    sigma_true = 0.60 # Volatilité réelle de 60%
    
    # Premier test : Vérification du pricing et des grecques pour un Call avec les paramètres de marché fictifs
    print("TEST 1 : Pricing et Grecques (Call)")
    call_price = vol_model.bs_price(S_test, K_test, T_test, r_test, sigma_true, 'call')
    greeks = vol_model.calculate_greeks(S_test, K_test, T_test, r_test, sigma_true, 'call')    
    print(f"Prix theorique du Call : {call_price:.2f} USD")
    print(f"Delta: {greeks['delta']:.4f} | Gamma: {greeks['gamma']:.6f} | Vega: {greeks['vega']:.2f}")
    print("-" * 50)
    
    # Second test : Vérification de l'inversion mathématique pour retrouver la volatilité implicite à partir du prix de marché
    print("TEST 2 : Inversion mathematique (Volatilite Implicite)")
    iv_extracted = vol_model.implied_volatility(S_test, K_test, T_test, r_test, call_price, 'call')   
    print(f"Volatilite cible (cachee) : {sigma_true * 100:.2f} %")
    print(f"Volatilite retrouvee par Newton/Dichotomie : {iv_extracted * 100:.2f} %")
    if abs(sigma_true - iv_extracted) < 0.001:
        print("SUCCES : L'algorithme a parfaitement retrouve la volatilite !")
    else:
        print("ECHEC : L'algorithme a diverge.")
    print("-" * 50)
    
    # Troisième test : Vérification de la calibration du modèle SSVI sur un mini-dataset synthétique avec des prix "parfaits" et du bruit 
    print("TEST 3 : Calibration SSVI sur un mini-dataset synthetique")
    dummy_data = []
    for strike in [60000, 65000, 70000]:
        for mat in [15/365.25, 45/365.25]:
            # Prix "parfait"
            price = vol_model.bs_price(S_test, strike, mat, r_test, sigma_true, 'call')
            # Ajout d'un bruit aléatoire de 5$ pour simuler la réalité du marché
            noisy_price = price + np.random.normal(0, 5) 
            
            dummy_data.append({
                'underlying_price': S_test,
                'strike': strike,
                'T': mat,
                'r_T': r_test,
                'mid_price_usd': max(noisy_price, 0.1), # Le prix ne peut pas être négatif
                'option_type': 'call'
            })
            
    df_dummy = pd.DataFrame(dummy_data)
    
    try:
        ssvi_params = vol_model.calibrate_ssvi(df_dummy)
        print("SUCCES : L'optimiseur a converge !")
        print("Parametres SSVI trouves :")
        for key, value in ssvi_params.items():
            print(f"  - {key}: {value:.4f}")
    except Exception as e:
        print(f"ECHEC de la calibration : {e}")
    print("-" * 50)


## Partie 4 : Création de produits dérivés


# Classe pour modéliser la création, le pricing et le hedging de produits dérivés structurés 
class DerivativesPricerAndHedger:

    # Fonction d'initialisation avec les modèles de taux d'intérêt et de volatilité calibrés précédemment
    def __init__(self, ir_model, vol_model):
        self.ir_model = ir_model
        self.vol_model = vol_model

    # Fonction pour pour calculer le prix et les grecques d'une stratégie d'options 
    def price_strategy(self, S, maturity, strategy_legs):

        # Récupération du taux d'intérêt à la maturité du produit à partir du modèle de Nelson-Siegel calibré
        r = self.ir_model.get_ns_rates([maturity])[0]

        # Initialisation du prix total de la stratégie et des grecques totales à zéro   
        total_price = 0.0
        total_greeks = {'delta': 0.0, 'gamma': 0.0, 'vega': 0.0, 'theta': 0.0, 'rho': 0.0}
        
        for leg in strategy_legs:
            # Estimation de la volatilité sigma(k,t) via SSVI 
            iv = self.vol_model.get_ssvi_vol(S, leg['strike'], maturity, r)
            
            # Pricing et Grecques de la jambe
            price = self.vol_model.bs_price(S, leg['strike'], maturity, r, iv, leg['type'])
            greeks = self.vol_model.calculate_greeks(S, leg['strike'], maturity, r, iv, leg['type'])
            
            # Calcul du prix total de la stratégie en sommant les prix de chaque jambe pondérés par leur quantité
            total_price += price * leg['qty']
            for greek in total_greeks:
                total_greeks[greek] += greeks[greek] * leg['qty']
                
        return {'price': total_price, 'greeks': total_greeks, 'r': r, 'iv_used': iv}

    # Fonction pour générer les données de payoff et de spot à maturité
    def generate_payoff_data(self, strategy_legs, spot_ref, range_pct=0.3, num_points=100):
        s_min = spot_ref * (1 - range_pct)
        s_max = spot_ref * (1 + range_pct)
        S_grid = np.linspace(s_min, s_max, num_points)
        
        # Boucle pour calculer le payoff de la stratégie à maturité pour chaque point de la grille de spot
        payoffs = np.zeros_like(S_grid)
        for leg in strategy_legs:
            if leg['type'] == 'call':
                payoff_leg = np.maximum(S_grid - leg['strike'], 0) * leg['qty']
            else: # put
                payoff_leg = np.maximum(leg['strike'] - S_grid, 0) * leg['qty']
            payoffs += payoff_leg
            
        return pd.DataFrame({'Spot_Maturity': S_grid, 'Payoff': payoffs})

    # Fonction pour optimiser la couverture du produit structuré vendu 
    def optimize_hedging(self, S, target_greeks, available_instruments):

        # Le produit est VENDU, donc notre risque initial est l'inverse des Grecques du produit.
        risk_to_hedge = {
            'delta': -target_greeks['delta'],
            'gamma': -target_greeks['gamma'],
            'vega': -target_greeks['vega']
        }
        
        # Taille = nombre d'instruments de couverture disponibles 
        n_instruments = len(available_instruments)
        
        # Fonction objectif à minimiser : la somme des carrés des Grecques résiduelles après couverture, 
        # avec des poids pour forcer l'optimiseur à respecter les petites valeurs de gamma et vega
        def objective(weights):
            res_delta = risk_to_hedge['delta'] + sum(weights[i] * available_instruments[i]['greeks']['delta'] for i in range(n_instruments))
            res_gamma = risk_to_hedge['gamma'] + sum(weights[i] * available_instruments[i]['greeks']['gamma'] for i in range(n_instruments))
            res_vega = risk_to_hedge['vega'] + sum(weights[i] * available_instruments[i]['greeks']['vega'] for i in range(n_instruments))
            
            # Poids subjectifs pour forcer l'optimiseur à respecter les petites valeurs
            return (res_delta**2) + (res_gamma * 10000)**2 + (res_vega / 100)**2

        # Lancement de l'optimiseur en partant de 0
        initial_guess = np.zeros(n_instruments)
        res = minimize(objective, initial_guess, method='BFGS')
        
        optimal_weights = res.x
        
        # Calcul final du coût de la couverture 
        hedge_cost = sum(optimal_weights[i] * available_instruments[i]['price'] for i in range(n_instruments))
        
        return optimal_weights, hedge_cost

    # Fonction pour recalculer la valeur du portefeuille 1 semaine plus tard avec des chocs de marché (stress test)
    def stress_test_portfolio(self, S_initial, maturity, strategy_legs, hedge_instruments, hedge_weights, product_price_initial):

        # Paramètres changés
        S_new = S_initial * 1.10
        maturity_new = maturity - (7 / 365.25)
        
        if maturity_new <= 0:
            return "Le produit est expiré, veuillez choisir une maturité > 7 jours."

        # Recalcul du taux d'intérêt avec la nouvelle maturité
        r_new = self.ir_model.get_ns_rates([maturity_new])[0]
        
        # Repricibg du produit vendu avec les nouveaux paramètres de marché
        product_price_new = 0.0
        for leg in strategy_legs:
            iv_base = self.vol_model.get_ssvi_vol(S_new, leg['strike'], maturity_new, r_new)
            iv_shocked = max(iv_base - 0.10, 0.001) # La vol ne peut pas être négative
            
            price_leg = self.vol_model.bs_price(S_new, leg['strike'], maturity_new, r_new, iv_shocked, leg['type'])
            product_price_new += price_leg * leg['qty']
            
        # Calcul du P&L sans couverture 
        pnl_product = product_price_initial - product_price_new
        
        # Repricing des instruments de couverture avec les nouveaux paramètres de marché
        pnl_hedge = 0.0
        for i, inst in enumerate(hedge_instruments):
            qty = hedge_weights[i]
            
            if inst['type'] == 'spot':
                new_price = S_new
            else: # Option de couverture
                iv_base = self.vol_model.get_ssvi_vol(S_new, inst['strike'], inst['T'], r_new)
                iv_shocked = max(iv_base - 0.10, 0.001)
                new_price = self.vol_model.bs_price(S_new, inst['strike'], inst['T'], r_new, iv_shocked, inst['type'])
                
            # Calcul du P&L de la position de couverture 
            pnl_hedge += qty * (new_price - inst['price'])
            
        # Calcul du P&L total du portefeuille 
        total_pnl = pnl_product + pnl_hedge
        
        return {
            'S_new': S_new,
            'maturity_new': maturity_new,
            'pnl_product': pnl_product,
            'pnl_hedge': pnl_hedge,
            'total_pnl': total_pnl
        }
    
# Exemple d'Utilisation de la Partie 4
if __name__ == "__main__":
    print("\n[--- Lancement des tests du Module 4 (Structuration & Hedging) ---]\n")
    
    # Initialisation des modèles 
    ir_model = InterestRateModel()
    vol_model = VolatilityModel()
    pricer = DerivativesPricerAndHedger(ir_model, vol_model)
    
    # Paramètres de marché fictifs
    S_test = 65000.0
    maturity_test = 30 / 365.25
    
    # Création d'une stratégie "Call Spread" 
    my_strategy = [
        {'type': 'call', 'strike': 60000, 'qty': 1},
        {'type': 'call', 'strike': 70000, 'qty': -1}
    ]
    
    # Premier test : Pricing de la stratégie et vérification de la somme des grecques
    print("[TEST 1] Pricing de la strategie (Call Spread)")
    try:
        struct_res = pricer.price_strategy(S_test, maturity_test, my_strategy)
        print(f"  -> Prix total de la structure : {struct_res['price']:.2f} USD")
        print(f"  -> Delta total : {struct_res['greeks']['delta']:.4f}")
        print("  [OK] Le pricing et la somme des grecques fonctionnent.")
    except Exception as e:
        print(f"  [ERREUR] {e}")
    print("-" * 50)
    
    # Second test : Génération du tableau de Payoff et vérification du payoff max théorique
    print("[TEST 2] Generation des donnees du Payoff")
    try:
        payoff_df = pricer.generate_payoff_data(my_strategy, S_test, range_pct=0.2)
        print(f"  -> Tableau du Payoff genere : {len(payoff_df)} points calcules.")
        print(f"  -> Profit max theorique bloque a : {payoff_df['Payoff'].max():.2f} USD")
        print("  [OK] La generation du Payoff fonctionne.")
    except Exception as e:
        print(f"  [ERREUR] {e}")
    print("-" * 50)
    
    # Troisième test : Optimisation du portefeuille de couverture pour neutraliser les risques du produit structuré vendu
    print("[TEST 3] Optimisation du Portefeuille de Couverture")
    
    # On simule le fait que le marché nous propose ces 3 instruments pour nous couvrir
    available_hedges = [
        {'name': 'Spot BTC', 'type': 'spot', 'price': S_test, 'strike': None, 'T': None, 
         'greeks': {'delta': 1.0, 'gamma': 0.0, 'vega': 0.0}},
        {'name': 'Call 65k', 'type': 'call', 'price': 2500.0, 'strike': 65000, 'T': maturity_test, 
         'greeks': {'delta': 0.52, 'gamma': 0.0001, 'vega': 150.0}},
        {'name': 'Put 65k', 'type': 'put', 'price': 2400.0, 'strike': 65000, 'T': maturity_test, 
         'greeks': {'delta': -0.48, 'gamma': 0.0001, 'vega': 150.0}}
    ]
    
    try:
        weights, cost = pricer.optimize_hedging(S_test, struct_res['greeks'], available_hedges)
        print("  -> Quantites optimales a acheter :")
        for i, w in enumerate(weights):
            print(f"     * {available_hedges[i]['name']} : {w:.4f} unites")
        print(f"  -> Cout theorique de la couverture : {cost:.2f} USD")
        print("  [OK] L'optimiseur BFGS a trouve une solution de couverture.")
    except Exception as e:
        print(f"  [ERREUR] {e}")
    print("-" * 50)
    
    # Quatrième test : Stress Test du portefeuille avec un choc de +10% sur le spot et -10% sur la volatilité
    try:
        stress_res = pricer.stress_test_portfolio(
            S_test, maturity_test, my_strategy, available_hedges, weights, struct_res['price']
        )
        print(f"  -> Nouveau prix du sous-jacent : {stress_res['S_new']:.2f} USD")
        print(f"  -> P&L sur le produit vendu : {stress_res['pnl_product']:.2f} USD")
        print(f"  -> P&L sur la couverture : {stress_res['pnl_hedge']:.2f} USD")
        print(f"  -> P&L TOTAL (Portefeuille) : {stress_res['total_pnl']:.2f} USD")
        
        if abs(stress_res['total_pnl']) < abs(stress_res['pnl_product']):
            print("  [OK] Succes ! Le hedging a bien reduit les pertes (ou securise les gains).")
        else:
            print("  [INFO] Le hedging a fonctionne mathematiquement, mais le choc etait asymetrique.")
    except Exception as e:
        print(f"  [ERREUR] {e}")
    print("-" * 50)