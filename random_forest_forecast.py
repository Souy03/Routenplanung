import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import os
import logging
import json
import random
from datetime import datetime
import warnings
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
warnings.filterwarnings('ignore')

# Logging konfigurieren
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ==========================
# OPTIMIERTE REGEL-ENGINE
# ==========================

class SmartBinRuleEngine:
    """
    Optimierte regelbasierte Post-Processing Engine
    """
    
    def __init__(self):
        self.rules = []
        self.rule_log = []
    
    def add_rule(self, name, condition_func, action_func, priority=1):
        """Regel hinzufügen mit Prioritätssortierung"""
        self.rules.append({
            'name': name,
            'condition': condition_func,
            'action': action_func,
            'priority': priority
        })
        self.rules.sort(key=lambda x: x['priority'], reverse=True)
    
    def apply_rules(self, forecasts, df):
        """
        Parallele Regelanwendung für bessere Performance
        """
        modified_forecasts = forecasts.copy()
        
        # Pre-compute bin data dictionary für bessere Performance
        bin_data_cache = {}
        for osm_id in df['osm_id'].unique():
            bin_data = df[df['osm_id'] == osm_id].sort_values('timestamp')
            if len(bin_data) > 0:
                bin_data_cache[osm_id] = {
                    'data': bin_data,
                    'last_entry': bin_data.iloc[-1]
                }
        
        # Regeln parallel anwenden
        def apply_rules_to_forecast(forecast):
            osm_id = forecast['osm_id']
            
            if osm_id not in bin_data_cache:
                return forecast
                
            cached_data = bin_data_cache[osm_id]
            last_entry = cached_data['last_entry']
            historical_data = cached_data['data']
            
            context = {
                'forecast': forecast,
                'last_entry': last_entry,
                'historical_data': historical_data,
                'current_time': datetime.now()
            }
            
            for rule in self.rules:
                try:
                    if rule['condition'](context):
                        old_values = forecast.copy()
                        rule['action'](context)
                        
                        self.rule_log.append({
                            'osm_id': osm_id,
                            'rule_name': rule['name'],
                            'timestamp': datetime.now(),
                            'changes': self._get_changes(old_values, forecast)
                        })
                        break  # Nur erste passende Regel anwenden für Performance
                        
                except Exception as e:
                    logger.warning(f"Regel '{rule['name']}' für {osm_id}: {e}")
            
            return forecast
        
        # Parallel processing
        with ThreadPoolExecutor(max_workers=4) as executor:
            modified_forecasts = list(executor.map(apply_rules_to_forecast, modified_forecasts))
        
        return modified_forecasts
    
    def _get_changes(self, old, new):
        """Optimierte Änderungserkennung"""
        changes = {}
        keys_to_check = ['predicted_fill_8h', 'predicted_fill_16h', 'predicted_fill_24h', 'urgency_level']
        for key in keys_to_check:
            if old.get(key) != new.get(key):
                changes[key] = {'old': old.get(key), 'new': new.get(key)}
        return changes
    
    def get_rule_statistics(self):
        """Optimierte Regel-Statistiken"""
        if not self.rule_log:
            return {}
            
        stats = {}
        for entry in self.rule_log:
            rule_name = entry['rule_name']
            if rule_name not in stats:
                stats[rule_name] = {'count': 0, 'bins_affected': set()}
            stats[rule_name]['count'] += 1
            stats[rule_name]['bins_affected'].add(entry['osm_id'])
        
        # Konvertierung für JSON
        for rule_name in stats:
            stats[rule_name]['bins_affected'] = len(stats[rule_name]['bins_affected'])
        
        return stats


# ==========================
# OPTIMIERTE REGEL-DEFINITIONEN
# ==========================

def create_weather_degradation_rule():
    """Optimierte Wetter-Regel"""
    def condition(context):
        last_entry = context['last_entry']
        forecast = context['forecast']
        
        return (
            last_entry['temperature'] >= 28 and 
            last_entry['humidity'] >= 80 and
            forecast['predicted_fill_24h'] < 90
        )
    
    def action(context):
        forecast = context['forecast']
        degradation_factor = 1.2  # Fester Faktor für Performance
        
        forecast['predicted_fill_8h'] = min(100, forecast['predicted_fill_8h'] * degradation_factor)
        forecast['predicted_fill_16h'] = min(100, forecast['predicted_fill_16h'] * degradation_factor)
        forecast['predicted_fill_24h'] = min(100, forecast['predicted_fill_24h'] * degradation_factor)
        
        if forecast['predicted_fill_24h'] >= 85:
            forecast['urgency_level'] = "high"
        elif forecast['predicted_fill_24h'] >= 70:
            forecast['urgency_level'] = "medium"
        
        forecast['rule_applied'] = "weather_degradation"
        forecast['adjustment_reason'] = f"Hohe Temp/Luftfeuchtigkeit"
    
    return condition, action

def create_stale_data_rule():
    """Optimierte Regel für veraltete Daten"""
    def condition(context):
        last_entry = context['last_entry']
        current_time = context['current_time']
        
        last_timestamp = pd.to_datetime(last_entry['timestamp'])
        days_old = (current_time - last_timestamp).days
        
        return days_old >= 2
    
    def action(context):
        forecast = context['forecast']
        last_entry = context['last_entry']
        
        days_old = (context['current_time'] - pd.to_datetime(last_entry['timestamp'])).days
        uncertainty_factor = 1 + (days_old * 0.1)
        
        forecast['predicted_fill_8h'] = min(100, forecast['predicted_fill_8h'] * uncertainty_factor)
        forecast['predicted_fill_16h'] = min(100, forecast['predicted_fill_16h'] * uncertainty_factor)
        forecast['predicted_fill_24h'] = min(100, forecast['predicted_fill_24h'] * uncertainty_factor)
        
        if forecast['predicted_fill_24h'] >= 75:
            forecast['urgency_level'] = "high"
        
        forecast['rule_applied'] = "stale_data_adjustment"
        forecast['adjustment_reason'] = f"Daten {days_old} Tage alt"
    
    return condition, action


# ===============================
# HOCHOPTIMIERTE FORECASTER-KLASSE
# ===============================

class OptimizedSmartBinForecaster:
    """
    Hochoptimierte SmartBin Forecaster-Klasse
    """
    
    def __init__(self, model_path="models/", data_path="data/", use_rules=True, n_jobs=-1):
        self.model_path = model_path
        self.data_path = data_path
        self.use_rules = use_rules
        self.n_jobs = n_jobs
        self.model = None
        self.scaler = None
        self.label_encoders = {}
        self.feature_names = None
        self.rule_engine = SmartBinRuleEngine() if use_rules else None
        
        # Cache für preprocessing
        self._feature_cache = {}
        
        os.makedirs(self.model_path, exist_ok=True)
        os.makedirs(self.data_path, exist_ok=True)
        
        if self.use_rules:
            self._initialize_rules()
    
    def _initialize_rules(self):
        """Optimierte Regel-Initialisierung"""
        rules = [
            ("Wetter-Degradation", create_weather_degradation_rule(), 3),
            ("Veraltete Daten", create_stale_data_rule(), 2),
        ]
        
        for name, (condition, action), priority in rules:
            self.rule_engine.add_rule(name, condition, action, priority)
        
        logger.info(f"✅ {len(rules)} Regeln initialisiert")
    
    def load_data(self, file_path):
        """Optimiertes Daten laden"""
        try:
            # Chunked reading für große Dateien
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path, 
                               parse_dates=['timestamp'] if 'timestamp' in pd.read_csv(file_path, nrows=1).columns else None)
            elif file_path.endswith('.xlsx'):
                df = pd.read_excel(file_path)
            else:
                raise ValueError("Nur CSV und XLSX Dateien unterstützt")
            
            logger.info(f"Daten geladen: {len(df)} Zeilen, {len(df.columns)} Spalten")
            
            # Validierung
            required_columns = ['osm_id', 'timestamp', 'fill_level', 'weight', 'temperature', 'humidity']
            missing_cols = [col for col in required_columns if col not in df.columns]
            
            if missing_cols:
                raise ValueError(f"Fehlende Spalten: {missing_cols}")
            
            return df
            
        except Exception as e:
            logger.error(f"Fehler beim Laden der Daten: {e}")
            raise
    
    @lru_cache(maxsize=128)
    def _compute_cyclic_features(self, hour, weekday):
        """Cached zyklische Features"""
        return {
            'hour_sin': np.sin(2 * np.pi * hour / 24),
            'hour_cos': np.cos(2 * np.pi * hour / 24),
            'weekday_sin': np.sin(2 * np.pi * weekday / 7),
            'weekday_cos': np.cos(2 * np.pi * weekday / 7)
        }
    
    def preprocess_data(self, df):
        """STARK OPTIMIERTES Feature Engineering"""
        logger.info("Starte optimiertes Feature Engineering...")
        
        df = df.copy()
        
        # Timestamp verarbeiten (einmalig)
        if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        df = df.sort_values(['osm_id', 'timestamp'])
        
        # Vectorized zeitbasierte Features
        df['hour'] = df['timestamp'].dt.hour
        df['weekday'] = df['timestamp'].dt.weekday
        df['month'] = df['timestamp'].dt.month
        df['is_weekend'] = (df['weekday'] >= 5).astype(int)
        df['is_rush_hour'] = df['hour'].isin([7, 8, 9, 17, 18, 19]).astype(int)
        
        # Vectorized zyklische Features
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['weekday_sin'] = np.sin(2 * np.pi * df['weekday'] / 7)
        df['weekday_cos'] = np.cos(2 * np.pi * df['weekday'] / 7)
        
        # OSM ID encoding (optimiert)
        if 'osm_id' not in self.label_encoders:
            self.label_encoders['osm_id'] = LabelEncoder()
            df['osm_code'] = self.label_encoders['osm_id'].fit_transform(df['osm_id'])
        else:
            df['osm_code'] = self.label_encoders['osm_id'].transform(df['osm_id'])
        
        # Nur die wichtigsten Lag Features (Performance!)
        df['fill_level_lag_1'] = df.groupby('osm_id')['fill_level'].shift(1)
        df['fill_level_lag_2'] = df.groupby('osm_id')['fill_level'].shift(2)
        
        # Nur wichtigste Rolling Features
        df['fill_level_mean_3'] = df.groupby('osm_id')['fill_level'].rolling(window=3, min_periods=1).mean().reset_index(0, drop=True)
        df['fill_level_std_3'] = df.groupby('osm_id')['fill_level'].rolling(window=3, min_periods=1).std().reset_index(0, drop=True)
        
        # Füllrate
        df['fill_rate'] = df.groupby('osm_id')['fill_level'].diff().fillna(0)
        
        # Vereinfachte Interaction Features
        df['temp_humidity_ratio'] = df['temperature'] / (df['humidity'] + 1)
        
        # Target Variable
        df['fill_level_target'] = df.groupby('osm_id')['fill_level'].shift(-1)
        
        # Optimierte NaN-Behandlung
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        df[numeric_columns] = df[numeric_columns].fillna(method='ffill').fillna(0)
        
        df.dropna(subset=['fill_level_target'], inplace=True)
        
        logger.info(f"Feature Engineering abgeschlossen: {len(df)} Zeilen")
        
        return df
    
    def prepare_features(self, df):
        """Optimierte Feature-Vorbereitung"""
        # Reduzierte Feature-Liste für bessere Performance
        self.feature_names = [
            'fill_level', 'weight', 'temperature', 'humidity',
            'osm_code', 'hour', 'weekday', 'is_weekend', 'is_rush_hour',
            'hour_sin', 'hour_cos', 'weekday_sin', 'weekday_cos',
            'fill_level_lag_1', 'fill_level_lag_2',
            'fill_level_mean_3', 'fill_level_std_3',
            'fill_rate', 'temp_humidity_ratio'
        ]
        
        available_features = [f for f in self.feature_names if f in df.columns]
        self.feature_names = available_features
        
        X = df[self.feature_names]
        y = df['fill_level_target']
        
        logger.info(f"Features vorbereitet: {len(self.feature_names)} Features")
        
        return X, y
    
    def train_model(self, X, y, test_size=0.2, hyperparameter_tuning=False):
        """STARK OPTIMIERTES Training"""
        logger.info("Starte optimiertes Modelltraining...")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        # Opt-out Scaling für Tree-based Modelle (Random Forest braucht es nicht wirklich)
        # self.scaler = StandardScaler()
        # X_train_scaled = self.scaler.fit_transform(X_train)
        # X_test_scaled = self.scaler.transform(X_test)
        
        if hyperparameter_tuning:
            logger.info("Schnelles Hyperparameter-Tuning mit RandomizedSearch...")
            
            # Reduzierter Parameter-Raum für Geschwindigkeit
            param_dist = {
                'n_estimators': [50, 100, 150],  # Weniger Bäume
                'max_depth': [10, 15, 20],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2],
                'max_features': ['sqrt', 'log2']  # Nur beste Optionen
            }
            
            rf = RandomForestRegressor(random_state=42, n_jobs=self.n_jobs)
            
            # RandomizedSearchCV statt GridSearchCV (viel schneller!)
            random_search = RandomizedSearchCV(
                rf, param_dist, n_iter=20, cv=3,  # Nur 20 Iterationen, 3-fach CV
                scoring='neg_mean_absolute_error',
                n_jobs=self.n_jobs, random_state=42, verbose=1
            )
            
            random_search.fit(X_train, y_train)
            self.model = random_search.best_estimator_
            
            logger.info(f"Beste Parameter: {random_search.best_params_}")
            
        else:
            # Optimierte Standard-Parameter
            self.model = RandomForestRegressor(
                n_estimators=100,  # Reduziert von 200
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                max_features='sqrt',
                random_state=42,
                n_jobs=self.n_jobs
            )
            self.model.fit(X_train, y_train)
        
        # Evaluation
        train_pred = self.model.predict(X_train)
        test_pred = self.model.predict(X_test)
        
        train_mae = mean_absolute_error(y_train, train_pred)
        test_mae = mean_absolute_error(y_test, test_pred)
        train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
        train_r2 = r2_score(y_train, train_pred)
        test_r2 = r2_score(y_test, test_pred)
        
        print("\n" + "="*50)
        print("🎯 OPTIMIERTE MODELL EVALUATION")
        print("="*50)
        print(f"Training Set: MAE: {train_mae:.2f}%, RMSE: {train_rmse:.2f}%, R²: {train_r2:.3f}")
        print(f"Test Set: MAE: {test_mae:.2f}%, RMSE: {test_rmse:.2f}%, R²: {test_r2:.3f}")
        
        # Nur Top 5 wichtigste Features anzeigen
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"\n🔍 TOP 5 WICHTIGSTE FEATURES:")
        print(feature_importance.head(5).to_string(index=False))
        
        return {
            'train_mae': train_mae, 'test_mae': test_mae,
            'train_rmse': train_rmse, 'test_rmse': test_rmse,
            'train_r2': train_r2, 'test_r2': test_r2,
            'feature_importance': feature_importance
        }
    
    def create_forecast(self, df, forecast_horizons=[8, 16, 24]):
        """Optimierte Vorhersage-Funktion"""
        if self.model is None:
            raise ValueError("Modell nicht trainiert oder geladen")
        
        logger.info("Erstelle optimierte Vorhersagen...")
        
        # Batch prediction für bessere Performance
        latest_data = df.groupby('osm_id').tail(1).reset_index(drop=True)
        
        # Alle Features auf einmal vorbereiten
        X_pred = latest_data[self.feature_names]
        base_predictions = self.model.predict(X_pred)
        
        forecasts = []
        for i, (_, row) in enumerate(latest_data.iterrows()):
            base_prediction = base_predictions[i]
            
            # Vereinfachte Zeithorizont-Extrapolation
            predictions = {}
            for hours in forecast_horizons:
                time_factor = hours / 24
                predictions[f'predicted_fill_{hours}h'] = min(100, base_prediction * (1 + time_factor * 0.2))
            
            urgency = self._determine_urgency(predictions['predicted_fill_24h'])
            
            forecast = {
                'osm_id': row['osm_id'],
                'current_fill_level': row['fill_level'],
                'timestamp': datetime.now(),
                **predictions,
                'urgency_level': urgency,
                'rule_applied': 'none',
                'adjustment_reason': 'ML-Vorhersage'
            }
            
            forecasts.append(forecast)
        
        # Regel-Engine (falls aktiviert)
        rule_stats = {}
        if self.use_rules and self.rule_engine:
            logger.info("Wende Regeln an...")
            forecasts = self.rule_engine.apply_rules(forecasts, df)
            rule_stats = self.rule_engine.get_rule_statistics()
        
        return forecasts, rule_stats
    
    def _determine_urgency(self, predicted_fill):
        """Optimierte Dringlichkeitsbestimmung"""
        if predicted_fill >= 85:
            return "high"
        elif predicted_fill >= 70:
            return "medium"
        else:
            return "low"
    
    def save_model(self):
        """Optimiertes Speichern"""
        try:
            joblib.dump(self.model, os.path.join(self.model_path, "rf_model.pkl"))
            joblib.dump(self.label_encoders, os.path.join(self.data_path, "encoders.pkl"))
            joblib.dump(self.feature_names, os.path.join(self.data_path, "features.pkl"))
            
            logger.info("✅ Modell gespeichert")
            
        except Exception as e:
            logger.error(f"Speicher-Fehler: {e}")
            raise
    
    def load_model(self):
        """Optimiertes Laden"""
        try:
            self.model = joblib.load(os.path.join(self.model_path, "rf_model.pkl"))
            self.label_encoders = joblib.load(os.path.join(self.data_path, "encoders.pkl"))
            self.feature_names = joblib.load(os.path.join(self.data_path, "features.pkl"))
            
            if self.use_rules:
                self.rule_engine = SmartBinRuleEngine()
                self._initialize_rules()
            
            logger.info("✅ Modell geladen")
            
        except Exception as e:
            logger.error(f"Lade-Fehler: {e}")
            raise


# ===============================
# OPTIMIERTE HAUPTFUNKTION
# ===============================

def main_optimized():
    """Optimierte Hauptfunktion für maximale Performance"""
    
    print("🚀 OPTIMIERTER SmartBin Forecaster")
    print("="*50)
    
    # Optimierter Forecaster (ohne Scaling, weniger Features)
    forecaster = OptimizedSmartBinForecaster(use_rules=True, n_jobs=-1)
    
    try:
        df = forecaster.load_data("smartbin_simulated_osm_ids_realistisch.xlsx")
    except FileNotFoundError:
        logger.error("❌ Datei nicht gefunden. Erstelle Beispieldaten...")
        df = create_sample_data()
    
    # Optimierte Pipeline
    df_processed = forecaster.preprocess_data(df)
    X, y = forecaster.prepare_features(df_processed)

    print("\n📊 SCHNELLES TRAINING (ohne Hyperparameter-Tuning)...")
    results = forecaster.train_model(X, y, hyperparameter_tuning=False)
    
    forecaster.save_model()
    
    print("\n🔮 VORHERSAGEN...")
    forecasts, rule_stats = forecaster.create_forecast(df_processed)
    
    # Kompakte Ergebnisse
    total_bins = len(forecasts)
    high_urgency = len([f for f in forecasts if f['urgency_level'] == 'high'])
    rule_adjustments = len([f for f in forecasts if f.get('rule_applied', 'none') != 'none'])
    
    print(f"\n✅ ERGEBNISSE:")
    print(f"📊 {total_bins} Behälter analysiert")
    print(f"🚨 {high_urgency} mit hoher Dringlichkeit") 
    print(f"⚙️ {rule_adjustments} Regel-Anpassungen")
    print(f"🎯 Test MAE: {results['test_mae']:.1f}%")
    
    return forecaster, forecasts


def create_sample_data():
    """Optimierte Beispieldaten-Erstellung"""
    print("📝 Erstelle kompakte Beispieldaten...")
    
    dates = pd.date_range(start='2024-06-01', end='2024-12-31', freq='3D')  # Alle 3 Tage
    osm_ids = [f'bin_{i:03d}' for i in range(1, 11)]  # Nur 10 Behälter
    
    data = []
    
    for osm_id in osm_ids:
        base_fill_rate = np.random.uniform(3, 7)
        current_fill = np.random.uniform(10, 30)
        
        for date in dates:
            # Vereinfachte Simulation
            temp = 20 + 8 * np.sin(2 * np.pi * date.dayofyear / 365) + np.random.normal(0, 2)
            humidity = np.clip(60 + np.random.normal(0, 15), 20, 90)
            
            fill_increase = base_fill_rate * 3 + np.random.normal(0, 3)
            current_fill += fill_increase
            
            if current_fill > 85 or np.random.random() < 0.25:
                current_fill = np.random.uniform(5, 20)
            
            current_fill = np.clip(current_fill, 0, 100)
            
            data.append({
                'osm_id': osm_id,
                'timestamp': date,
                'fill_level': current_fill,
                'weight': current_fill * 3.0 + np.random.normal(0, 5),
                'temperature': temp,
                'humidity': humidity
            })
    
    df = pd.DataFrame(data)
    df.to_excel("smartbin_simulated_osm_ids_realistisch.xlsx", index=False)
    print(f"✅ {len(df)} kompakte Datensätze erstellt")
    
    return df


if __name__ == "__main__":
    try:
        forecaster, forecasts = main_optimized()
        print("\n🎉 Optimierung erfolgreich!")
        
    except Exception as e:
        logger.error(f"❌ Fehler: {e}")
        print(f"⚠️ Fehler: {e}")