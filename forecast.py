# 🔧 Standardbibliotheken und ML-Tools importieren
import pandas as pd
import numpy as np
import joblib
import os
import folium
import json
import geopandas as gpd
from datetime import datetime, timedelta
import logging
import warnings
import holidays
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder
from concurrent.futures import ThreadPoolExecutor
from shapely.geometry import shape, Point


# Warnungen unterdrücken (z. B. bei NaN)
warnings.filterwarnings('ignore')

# Logging konfigurieren
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Regel-Engine für Prognoseanpassungen
class SmartBinRuleEngine:
    def __init__(self):
        self.rules = []          # Liste aller Regeln (mit Prioritäten)
        self.rule_log = []       # Liste angewendeter Regeln mit Zeitstempel und Änderungen

    def add_rule(self, name, condition_func, action_func, priority=1):
        self.rules.append({
            'name': name,
            'condition': condition_func,
            'action': action_func,
            'priority': priority
        })
        self.rules.sort(key=lambda x: x['priority'], reverse=True)

    def apply_rules(self, forecasts, df):
        modified_forecasts = forecasts.copy()
        bin_data_cache = {}

        for WasteBasketID in df['WasteBasketID'].unique():
            bin_data = df[df['WasteBasketID'] == WasteBasketID].sort_values('timestamp')
            if len(bin_data) > 0:
                bin_data_cache[WasteBasketID] = {
                    'data': bin_data,
                    'last_entry': bin_data.iloc[-1]
                }

        def apply_rules_to_forecast(forecast):
            WasteBasketID = forecast['WasteBasketID']
            if WasteBasketID not in bin_data_cache:
                return forecast
            cached_data = bin_data_cache[WasteBasketID]
            last_entry = cached_data['last_entry']

            context = {
                'forecast': forecast,
                'last_entry': last_entry,
                'historical_data': cached_data['data'],
                'current_time': datetime.now()
            }

            for rule in self.rules:
                try:
                    if rule['condition'](context):
                        old_values = forecast.copy()
                        rule['action'](context)
                        self.rule_log.append({
                            'WasteBasketID': WasteBasketID,
                            'rule_name': rule['name'],
                            'timestamp': datetime.now(),
                            'changes': self._get_changes(old_values, forecast)
                        })
                        break
                except Exception as e:
                    logger.warning(f"Regel '{rule['name']}' für {WasteBasketID}: {e}")
            return forecast

        with ThreadPoolExecutor(max_workers=4) as executor:
            modified_forecasts = list(executor.map(apply_rules_to_forecast, modified_forecasts))

        return modified_forecasts

    def _get_changes(self, old, new):
        changes = {}
        for key in ['predicted_fill_8h', 'predicted_fill_16h', 'predicted_fill_24h', 'urgency_level']:
            if old.get(key) != new.get(key):
                changes[key] = {'old': old.get(key), 'new': new.get(key)}
        return changes

    def get_rule_statistics(self):
        stats = {}
        for entry in self.rule_log:
            rule_name = entry['rule_name']
            if rule_name not in stats:
                stats[rule_name] = {'count': 0, 'bins_affected': set()}
            stats[rule_name]['count'] += 1
            stats[rule_name]['bins_affected'].add(entry['WasteBasketID'])

        for rule_name in stats:
            stats[rule_name]['bins_affected'] = len(stats[rule_name]['bins_affected'])

        return stats

# 🌡️ Regel 1: Wetterbedingte Verschlechterung
def create_weather_degradation_rule():
    def condition(context):
        return (
            context['last_entry']['temperature'] >= 28 and
            context['last_entry']['humidity'] >= 80 and
            context['forecast']['predicted_fill_24h'] < 90
        )

    def action(context):
        forecast = context['forecast']
        for h in [8, 16, 24]:
            forecast[f'predicted_fill_{h}h'] = min(100, forecast[f'predicted_fill_{h}h'] * 1.2)

        if forecast['predicted_fill_24h'] >= 85:
            forecast['urgency_level'] = "high"
        elif forecast['predicted_fill_24h'] >= 70:
            forecast['urgency_level'] = "medium"

        forecast['rule_applied'] = "weather_degradation"
        forecast['adjustment_reason'] = "Hohe Temp/Luftfeuchtigkeit"

    return condition, action

# ⌛ Regel 2: Veraltete Daten
def create_stale_data_rule():
    def condition(context):
        return (context['current_time'] - pd.to_datetime(context['last_entry']['timestamp'])).days >= 2

    def action(context):
        forecast = context['forecast']
        days_old = (context['current_time'] - pd.to_datetime(context['last_entry']['timestamp'])).days
        factor = 1 + (days_old * 0.1)
        for h in [8, 16, 24]:
            forecast[f'predicted_fill_{h}h'] = min(100, forecast[f'predicted_fill_{h}h'] * factor)

        if forecast['predicted_fill_24h'] >= 75:
            forecast['urgency_level'] = "high"

        forecast['rule_applied'] = "stale_data_adjustment"
        forecast['adjustment_reason'] = f"Daten {days_old} Tage alt"

    return condition, action

# 📆 Regel 3: Feiertage und Wochenenden in Bayern

def create_holiday_weekend_rule():
    def condition(context):
        ts = pd.to_datetime(context['last_entry']['timestamp'])
        bavarian_holidays = holidays.Germany(prov="BY", years=ts.year)
        is_weekend = ts.weekday() >= 5
        is_holiday = ts in bavarian_holidays
        return is_weekend or is_holiday

    def action(context):
        forecast = context['forecast']
        for h in [8, 16, 24]:
            forecast[f'predicted_fill_{h}h'] = min(100, forecast[f'predicted_fill_{h}h'] * 1.15)
        forecast['rule_applied'] = "holiday_or_weekend"
        forecast['adjustment_reason'] = "Feiertag oder Wochenende in Bayern"

    return condition, action

# 📍 Regel 4: Standortregel mit Hotspot-Tags (basierend auf GeoJSON-Eigenschaft "hotspot_tags")
def create_location_hotspot_rule():
    def condition(context):
        tags = context['last_entry'].get('hotspot_tags', [])
        return bool(tags)

    def action(context):
        forecast = context['forecast']
        for h in [8, 16, 24]:
            forecast[f'predicted_fill_{h}h'] = min(100, forecast[f'predicted_fill_{h}h'] * 1.25)
        forecast['rule_applied'] = "location_hotspot"
        forecast['adjustment_reason'] = "Standort in Hotspot-Zone"

    return condition, action

# Sensorfehler Regel
def create_sensor_error_rule():
    def condition(context):
        fill_level = context['last_entry'].get('fill_level')
        # Regel greift bei fehlendem oder zu hohem Füllstand
        return fill_level is None or pd.isna(fill_level) or fill_level > 100

    def action(context):
        fill_level = context['last_entry'].get('fill_level')
        if fill_level is not None and not pd.isna(fill_level):
            for h in [8, 16, 24]:
                context['forecast'][f'predicted_fill_{h}h'] = min(100, fill_level)
        context['forecast']['rule_applied'] = 'sensor_error'
        context['forecast']['adjustment_reason'] = 'Füllstand fehlt oder > 100%'

    return condition, action


class OptimizedSmartBinForecaster:
    def __init__(self, model_path="models/", use_rules=True, n_jobs=-1):
        self.model_path = model_path
        self.model = None
        self.label_encoders = {}
        self.feature_names = []
        self.n_jobs = n_jobs
        self.use_rules = use_rules
        self.rule_engine = SmartBinRuleEngine() if use_rules else None
        os.makedirs(self.model_path, exist_ok=True)

        if self.use_rules:
            self._initialize_rules()

    def _initialize_rules(self):
        # Regeln mit unterschiedlichen Prioritäten hinzufügen
        self.rule_engine.add_rule("Wetteranpassung", *create_weather_degradation_rule(), priority=3)
        self.rule_engine.add_rule("Veraltete Daten", *create_stale_data_rule(), priority=2)
        self.rule_engine.add_rule("Standort-Hotspot", *create_location_hotspot_rule(), priority=2)
        self.rule_engine.add_rule("Feiertag", *create_holiday_weekend_rule(), priority=2)
        self.rule_engine.add_rule("Sensorfehler", *create_sensor_error_rule(), priority=1)

    def preprocess_data(self, df):
        df = df.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.sort_values(['WasteBasketID', 'timestamp'], inplace=True)

        # Zeitbasierte Features
        df['hour'] = df['timestamp'].dt.hour
        df['weekday'] = df['timestamp'].dt.weekday
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['weekday_sin'] = np.sin(2 * np.pi * df['weekday'] / 7)
        df['weekday_cos'] = np.cos(2 * np.pi * df['weekday'] / 7)

        # Rolling Fill Features
        df['fill_level_lag_1'] = df.groupby('WasteBasketID')['fill_level'].shift(1)
        df['fill_level_lag_2'] = df.groupby('WasteBasketID')['fill_level'].shift(2)
        df['fill_level_mean_3'] = df.groupby('WasteBasketID')['fill_level'].rolling(3, min_periods=1).mean().reset_index(0, drop=True)
        df['fill_level_std_3'] = df.groupby('WasteBasketID')['fill_level'].rolling(3, min_periods=1).std().reset_index(0, drop=True)
        df['fill_rate'] = df.groupby('WasteBasketID')['fill_level'].diff().fillna(0)

        # Temperaturverhältnis
        df['temp_humidity_ratio'] = df['temperature'] / (df['humidity'] + 1)

        # Zielwert definieren (z. B. Füllstand in 1 Zeiteinheit)
        df['fill_level_target'] = df.groupby('WasteBasketID')['fill_level'].shift(-1)

        # Fehlende Werte behandeln
        df.fillna(method='ffill', inplace=True)
        df.fillna(0, inplace=True)
        df.dropna(subset=['fill_level_target'], inplace=True)

        return df

    def load_and_preprocess_geojson(self, geojson_path: str) -> pd.DataFrame:
        """
        Lädt die GeoJSON-Datei und erstellt ein DataFrame im erwarteten Format
        mit Dummy-Zeitstempeln (z. B. 48h Verlauf im Stundenabstand).
        """
        gdf = gpd.read_file(geojson_path)

        if "id" not in gdf.columns:
            raise ValueError("Die GeoJSON-Datei muss eine 'id'-Spalte enthalten (Sensor-ID).")

        # Simuliere Zeitverlauf für die letzten 48 Stunden
        timestamps = pd.date_range(end=datetime.now(), periods=48, freq="H")
        rows = []
        for _, row in gdf.iterrows():
            for ts in timestamps:
                rows.append({
                    "sensor_id": str(row["id"]),
                    "latitude": row.geometry.y,
                    "longitude": row.geometry.x,
                    "timestamp": ts,
                    # Dummy-Füllstand (optional: durch Zufallszahlen oder echten Sensorwert ersetzen)
                    "fill_level": None
                })
        df = pd.DataFrame(rows)
        return df

    def prepare_features(self, df):
        features = [
            'fill_level', 'temperature', 'humidity', 'hour', 'weekday',
            'hour_sin', 'hour_cos', 'weekday_sin', 'weekday_cos',
            'fill_level_lag_1', 'fill_level_lag_2', 'fill_level_mean_3', 'fill_level_std_3',
            'fill_rate', 'temp_humidity_ratio'
        ]
        self.feature_names = [f for f in features if f in df.columns]
        return df[self.feature_names], df['fill_level_target']

    def train_model(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.model = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=self.n_jobs)
        self.model.fit(X_train, y_train)

        print("Train MAE:", mean_absolute_error(y_train, self.model.predict(X_train)))
        print("Test MAE:", mean_absolute_error(y_test, self.model.predict(X_test)))
        print("Test R2:", r2_score(y_test, self.model.predict(X_test)))

    def save_model(self):
        joblib.dump({
            'model': self.model,
            'features': self.feature_names
        }, os.path.join(self.model_path, "smartbin_rf_model.pkl"))

    def load_model(self):
        model_data = joblib.load(os.path.join(self.model_path, "smartbin_rf_model.pkl"))
        self.model = model_data['model']
        self.feature_names = model_data['features']

    def create_forecast(self, df, forecast_horizons=[8, 16, 24]):
        if self.model is None:
            raise RuntimeError("Kein Modell geladen oder trainiert")

        latest_data = df.groupby('WasteBasketID').tail(1).reset_index(drop=True)
        X_pred = latest_data[self.feature_names]
        base_predictions = self.model.predict(X_pred)

        forecasts = []
        for i, row in latest_data.iterrows():
            base_prediction = base_predictions[i]
            pred = {f'predicted_fill_{h}h': min(100, base_prediction * (1 + h / 24 * 0.2)) for h in forecast_horizons}
            urgency = "high" if pred['predicted_fill_24h'] >= 80 else "medium" if pred['predicted_fill_24h'] >= 60 else "low"
            forecast = {
                'WasteBasketID': row['WasteBasketID'],
                'timestamp': datetime.now(),
                'current_fill_level': row['fill_level'],
                **pred,
                'urgency_level': urgency,
                'rule_applied': "none",
                'location': row.get('location', '')
            }
            forecasts.append(forecast)

        if self.use_rules and self.rule_engine:
            forecasts = self.rule_engine.apply_rules(forecasts, df)

        return forecasts


def train_dummy_model_from_geojson(geojson_path: str, model_path: str = "models/"):
        """
        Generiert Dummy-Daten mit echten IDs aus GeoJSON und trainiert ein RandomForest-Modell.
        """
        gdf = gpd.read_file(geojson_path)
        if "id" not in gdf.columns:
            raise ValueError("GeoJSON muss eine 'id'-Spalte enthalten (Sensor-IDs).")

        rows = []
        now = datetime.now()

        for _, row in gdf.iterrows():
            bin_id = str(row["id"])
            fill_level = np.random.randint(10, 40)
            for h in range(72):
                timestamp = now - timedelta(hours=72 - h)
                temp = np.random.uniform(18, 35)
                humidity = np.random.uniform(50, 95)
                fill_level += np.random.uniform(0, 2)
                fill_level = min(fill_level, 100)
                rows.append({
                    "WasteBasketID": bin_id,
                    "timestamp": timestamp,
                    "fill_level": fill_level,
                    "temperature": temp,
                    "humidity": humidity
                })

        raw_df = pd.DataFrame(rows)
        forecaster = OptimizedSmartBinForecaster(model_path=model_path, use_rules=True)
        df = forecaster.preprocess_data(raw_df)
        X, y = forecaster.prepare_features(df)
        forecaster.train_model(X, y)
        forecaster.save_model()
        print("✅ Dummy-Modell trainiert und gespeichert unter:", model_path)

if __name__ == "__main__":
    # Nur einmal Dummy-Modell erzeugen
    train_dummy_model_from_geojson("waste_baskets_with_hotspots.geojson")


