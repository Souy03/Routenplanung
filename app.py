from flask import Flask, render_template, request, redirect, url_for, send_file, jsonify
import pandas as pd
import numpy as np
import pickle, torch, io, os, json
from datetime import datetime, timedelta
import random
import logging
from werkzeug.utils import secure_filename
import joblib
from sklearn.preprocessing import StandardScaler

# Logging konfigurieren
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'

# Upload Ordner erstellen
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs("static", exist_ok=True)

class ModelManager:
    """
    Random Forest Modellverwaltung
    """
    
    def __init__(self):
        self.rf_model = None
        self.scaler = None
        self.label_encoders = None
        self.feature_names = None
        
        # Modell laden
        self.load_model()
    
    def load_model(self):
        """
        Random Forest Modell laden
        """
        try:
            self.rf_model = joblib.load("models/random_forest_model.pkl")
            self.scaler = joblib.load("data/scaler.pkl")
            logger.info("✅ Random Forest Modell erfolgreich geladen")
            self.feature_names = [
    'fill_level', 'weight', 'temperature', 'humidity',
    'osm_code', 'hour', 'weekday', 'month', 'is_weekend',
    'is_rush_hour', 'hour_sin', 'hour_cos',
    'weekday_sin', 'weekday_cos',
    'fill_level_lag_1', 'fill_level_lag_2',
    'fill_rate', 'temp_humidity_ratio', 'fill_weight_ratio'
]

        except FileNotFoundError as e:
            logger.error(f"❌ Fehler beim Laden des Random Forest Modells: {e}")
            raise FileNotFoundError("Random Forest Modell nicht gefunden. Bitte stellen Sie sicher, dass alle Modelldateien vorhanden sind.")
    
    def is_available(self):
        """
        Prüft ob das Modell verfügbar ist
        """
        return self.rf_model is not None

# Global Model Manager
model_manager = ModelManager()

class DataProcessor:
    """
    Datenverarbeitung für Random Forest Vorhersagen
    """
    
    @staticmethod
    def validate_data(df):
        """
        Datenvalidierung mit detailliertem Feedback
        """
        errors = []
        warnings = []
        
        # Pflichtfelder prüfen
        required_columns = ['osm_id', 'timestamp', 'fill_level', 'weight', 'temperature', 'humidity']
        missing_cols = [col for col in required_columns if col not in df.columns]
        
        if missing_cols:
            errors.append(f"Fehlende Spalten: {', '.join(missing_cols)}")
        
        # Datentypen prüfen
        if 'timestamp' in df.columns:
            try:
                pd.to_datetime(df['timestamp'])
            except:
                errors.append("Timestamp-Spalte kann nicht als Datum interpretiert werden")
        
        # Wertebereiche prüfen
        if 'fill_level' in df.columns:
            invalid_fill = df[(df['fill_level'] < 0) | (df['fill_level'] > 100)]
            if len(invalid_fill) > 0:
                warnings.append(f"{len(invalid_fill)} Füllstände außerhalb 0-100%")
        
        if 'temperature' in df.columns:
            extreme_temps = df[(df['temperature'] < -20) | (df['temperature'] > 50)]
            if len(extreme_temps) > 0:
                warnings.append(f"{len(extreme_temps)} extreme Temperaturwerte")
        
        # Duplikate prüfen
        if 'osm_id' in df.columns and 'timestamp' in df.columns:
            duplicates = df.duplicated(subset=['osm_id', 'timestamp'])
            if duplicates.sum() > 0:
                warnings.append(f"{duplicates.sum()} doppelte Zeitstempel pro Behälter")
        
        return errors, warnings
    
    @staticmethod
    def preprocess_for_prediction(df):
        """
        Preprocessing für Random Forest Vorhersagen
        """
        df = df.copy()
        
        # Timestamp verarbeiten
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values(['osm_id', 'timestamp'])
        
        # Grundlegende zeitliche Features
        df['hour'] = df['timestamp'].dt.hour
        df['weekday'] = df['timestamp'].dt.weekday
        df['month'] = df['timestamp'].dt.month
        df['is_weekend'] = df['weekday'].isin([5, 6]).astype(int)
        df['is_rush_hour'] = df['hour'].isin([7, 8, 9, 17, 18, 19]).astype(int)
        
        # Zyklische Encoding
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['weekday_sin'] = np.sin(2 * np.pi * df['weekday'] / 7)
        df['weekday_cos'] = np.cos(2 * np.pi * df['weekday'] / 7)
        
        # OSM ID Encoding (numerisch)
        unique_ids = df['osm_id'].unique()
        id_mapping = {id_val: idx for idx, id_val in enumerate(unique_ids)}
        df['osm_code'] = df['osm_id'].map(id_mapping)
        
        # Lag Features
        for lag in [1, 2]:
            df[f'fill_level_lag_{lag}'] = df.groupby('osm_id')['fill_level'].shift(lag)
        
        # Füllrate
        df['fill_rate'] = df.groupby('osm_id')['fill_level'].diff().fillna(0)
        
        # Interaction Features
        df['temp_humidity_ratio'] = df['temperature'] / (df['humidity'] + 1)
        df['fill_weight_ratio'] = df['fill_level'] / (df['weight'] + 1)
        
        # NaN behandeln
        df = df.fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        return df

def create_forecast_rf(df):
    """
    Random Forest basierte Vorhersage
    """
    if not model_manager.is_available():
        raise ValueError("Random Forest Modell nicht verfügbar")

    df_processed = DataProcessor.preprocess_for_prediction(df)
    forecasts = []

    for osm_id, group in df_processed.groupby('osm_id'):
        if len(group) < 3:
            logger.warning(f"Zu wenig Daten für OSM ID {osm_id} (nur {len(group)} Datenpunkte)")
            continue

        last_row = group.iloc[-1].copy()
        predictions = []

        for hours_ahead in [8, 16, 24]:
            future_time = last_row['timestamp'] + pd.Timedelta(hours=hours_ahead)
            future_row = last_row.copy()
            future_row['timestamp'] = future_time
            future_row['hour'] = future_time.hour
            future_row['weekday'] = future_time.weekday()
            future_row['is_weekend'] = int(future_row['weekday'] in [5, 6])
            future_row['is_rush_hour'] = int(future_row['hour'] in [7, 8, 9, 17, 18, 19])
            future_row['hour_sin'] = np.sin(2 * np.pi * future_row['hour'] / 24)
            future_row['hour_cos'] = np.cos(2 * np.pi * future_row['hour'] / 24)
            future_row['weekday_sin'] = np.sin(2 * np.pi * future_row['weekday'] / 7)
            future_row['weekday_cos'] = np.cos(2 * np.pi * future_row['weekday'] / 7)
            future_row['temp_humidity_ratio'] = future_row['temperature'] / (future_row['humidity'] + 1e-6)
            future_row['fill_weight_ratio'] = future_row['fill_level'] / (future_row['weight'] + 1e-6)

            # Features für Vorhersage vorbereiten
            try:
                feature_df = pd.DataFrame([future_row])[model_manager.feature_names]
                X_scaled = model_manager.scaler.transform(feature_df)
                predicted_fill = model_manager.rf_model.predict(X_scaled)[0]
                
                # Sicherstellen, dass Vorhersage im gültigen Bereich liegt
                predicted_fill = max(0, min(100, predicted_fill))
                predictions.append(round(predicted_fill, 1))
                
            except Exception as e:
                logger.error(f"Fehler bei Vorhersage für OSM ID {osm_id}, {hours_ahead}h: {e}")
                # Fallback: Linearer Trend basierend auf letzten Werten
                if len(group) >= 2:
                    recent_trend = group['fill_level'].iloc[-1] - group['fill_level'].iloc[-2]
                    fallback_pred = last_row['fill_level'] + (recent_trend * hours_ahead / 6)
                    predictions.append(round(max(0, min(100, fallback_pred)), 1))
                else:
                    predictions.append(round(last_row['fill_level'], 1))

        # Dringlichkeit bestimmen
        max_fill = max(predictions)
        if max_fill >= 85:
            urgency = "high"
        elif max_fill >= 70:
            urgency = "medium"
        else:
            urgency = "low"

        forecasts.append({
            "osm_id": osm_id,
            "current_fill": round(last_row['fill_level'], 1),
            "predicted_fill_8h": predictions[0],
            "predicted_fill_16h": predictions[1],
            "predicted_fill_24h": predictions[2],
            "urgency_level": urgency,
            "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M")
        })

    return forecasts

def load_data_from_geojson(geojson_path):
    """
    Lädt Sensordaten basierend auf GeoJSON (falls verfügbar) oder erstellt Beispieldaten
    """
    try:
        with open(geojson_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        features = data.get("features", [])[:15]  # Maximal 15 Behälter
        logger.info(f"GeoJSON geladen: {len(features)} Behälter gefunden")
        
    except FileNotFoundError:
        logger.warning(f"GeoJSON-Datei nicht gefunden: {geojson_path}")
        # Erstelle einfache Beispiel-IDs
        features = [{"id": f"example_bin_{i}"} for i in range(1, 8)]
    
    start_date = datetime.now() - timedelta(days=2)
    data_rows = []
    
    for i, feature in enumerate(features):
        osm_id = feature.get("@id") or feature.get("id") or f"bin_{i}"
        
        # Realistische Füllstand-Entwicklung
        current_fill = random.uniform(10, 35)
        
        # 48 Stunden mit 6-stündigen Intervallen
        for hour_offset in range(0, 49, 6):
            timestamp = start_date + timedelta(hours=hour_offset)
            
            # Tageszeit-abhängige Füllrate
            hour = timestamp.hour
            if 6 <= hour <= 10:  # Morgens
                fill_increase = random.uniform(3, 8)
            elif 11 <= hour <= 14:  # Mittags
                fill_increase = random.uniform(8, 15)
            elif 15 <= hour <= 18:  # Nachmittags
                fill_increase = random.uniform(5, 12)
            elif 19 <= hour <= 22:  # Abends
                fill_increase = random.uniform(6, 10)
            else:  # Nacht
                fill_increase = random.uniform(0, 3)
            
            current_fill += fill_increase
            
            # Leerung simulieren
            if current_fill > 90 or (current_fill > 75 and random.random() < 0.3):
                current_fill = random.uniform(5, 20)
            
            # Sensordaten generieren
            weight = current_fill * random.uniform(0.35, 0.5)
            temp = random.uniform(-5, 30)
            humidity = max(20, min(95, 70 + random.uniform(-25, 25)))
            
            data_rows.append({
                "osm_id": osm_id,
                "timestamp": timestamp,
                "fill_level": round(current_fill, 1),
                "weight": round(weight, 2),
                "temperature": round(temp, 1),
                "humidity": round(humidity, 1)
            })
    
    return pd.DataFrame(data_rows)

# --- Flask Routen ---

@app.route("/")
def home():
    """
    Startseite mit Random Forest Modell-Info
    """
    model_info = {
        'type': 'Random Forest',
        'available': model_manager.is_available(),
        'features': len(model_manager.feature_names) if model_manager.feature_names else 0
    }
    return render_template("index.html", model_info=model_info)

@app.route("/api/health")
def health_check():
    """
    API Endpoint für Gesundheitsprüfung
    """
    return jsonify({
        "status": "healthy" if model_manager.is_available() else "model_unavailable",
        "model_type": "Random Forest",
        "model_available": model_manager.is_available(),
        "timestamp": datetime.now().isoformat()
    })

@app.route("/upload", methods=["POST"])
def upload():
    """
    Upload-Funktionalität nur für Random Forest
    """
    if not model_manager.is_available():
        return render_template("error.html", 
                             error="Random Forest Modell nicht verfügbar. Bitte überprüfen Sie die Modelldateien."), 500

    try:
        source = request.form.get("source")
        
        if source == "file":
            file = request.files.get("file")
            if not file or file.filename == '':
                return render_template("error.html", 
                                     error="Keine Datei ausgewählt"), 400

            # Sichere Dateinamen
            filename = secure_filename(file.filename)
            if not filename.lower().endswith(('.csv', '.xlsx')):
                return render_template("error.html", 
                                     error="Nur CSV- und XLSX-Dateien werden unterstützt"), 400

            # Datei temporär speichern
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            try:
                if filename.lower().endswith(".csv"):
                    df = pd.read_csv(filepath, encoding='utf-8')
                else:
                    df = pd.read_excel(filepath)
                
                # Temporäre Datei löschen
                os.remove(filepath)
                
            except UnicodeDecodeError:
                # Fallback für verschiedene Encodings
                df = pd.read_csv(filepath, encoding='latin1')
                os.remove(filepath)

        elif source == "geojson":
            df = load_data_from_geojson("geo/waste_baskets_nbg.geojson")
        
        else:
            return render_template("error.html", 
                                 error="Ungültige Datenquelle ausgewählt"), 400

        # Datenvalidierung
        errors, warnings = DataProcessor.validate_data(df)
        
        if errors:
            return render_template("error.html", 
                                 error=f"Datenvalidierung fehlgeschlagen: {'; '.join(errors)}"), 400

        if len(df) == 0:
            return render_template("error.html", 
                                 error="Datei enthält keine verwertbaren Daten"), 400

        # Random Forest Vorhersage durchführen
        results = create_forecast_rf(df)
        
        if not results:
            return render_template("error.html", 
                                 error="Keine Vorhersagen möglich. Prüfen Sie Ihre Daten und stellen Sie sicher, dass genügend historische Daten vorhanden sind."), 400

        df_results = pd.DataFrame(results)

        # Ergebnisse speichern
        output_path = "static/latest_forecast.csv"
        df_results.to_csv(output_path, index=False)
        
        # Statistiken
        stats = {
            'total_bins': len(results),
            'high_urgency': len([r for r in results if r['urgency_level'] == 'high']),
            'medium_urgency': len([r for r in results if r['urgency_level'] == 'medium']),
            'low_urgency': len([r for r in results if r['urgency_level'] == 'low']),
            'avg_current_fill': round(df_results['current_fill'].mean(), 1),
            'max_predicted_24h': round(df_results['predicted_fill_24h'].max(), 1),
            'warnings': warnings
        }

        return render_template("result.html", 
                             tables=[df_results.to_html(classes="table table-striped", 
                                                       table_id="forecast-table", 
                                                       index=False, escape=False)],
                             stats=stats,
                             model_type="Random Forest")

    except Exception as e:
        logger.error(f"Fehler beim Upload: {str(e)}")
        return render_template("error.html", 
                             error=f"Unerwarteter Fehler: {str(e)}"), 500

@app.route("/dashboard")
def show_dashboard():
    """
    Dashboard für Random Forest Ergebnisse
    """
    try:
        path = "static/latest_forecast.csv"
        if not os.path.exists(path):
            return render_template("error.html", 
                                 error="Noch keine Prognose verfügbar. Bitte laden Sie zuerst Daten hoch."), 400

        df = pd.read_csv(path)
        records = df.to_dict(orient="records")
        
        # Statistiken
        stats = {
            'total_bins': len(records),
            'high_urgency': len([r for r in records if r['urgency_level'] == 'high']),
            'medium_urgency': len([r for r in records if r['urgency_level'] == 'medium']),
            'low_urgency': len([r for r in records if r['urgency_level'] == 'low']),
            'avg_current_fill': round(df['current_fill'].mean(), 1),
            'max_current_fill': round(df['current_fill'].max(), 1),
            'avg_predicted_24h': round(df['predicted_fill_24h'].mean(), 1),
            'bins_full_24h': len(df[df['predicted_fill_24h'] >= 90]),
            'model_type': 'Random Forest'
        }
        
        # Dringlichkeitsliste
        urgent_bins = sorted(records, key=lambda x: x['predicted_fill_24h'], reverse=True)[:10]
        
        return render_template("dashboard.html", 
                             data=records, 
                             stats=stats, 
                             urgent_bins=urgent_bins)
        
    except Exception as e:
        logger.error(f"Fehler im Dashboard: {str(e)}")
        return render_template("error.html", 
                             error=f"Fehler beim Laden des Dashboards: {str(e)}"), 500

@app.route("/api/forecast")
def api_forecast():
    """
    API Endpoint für Vorhersagedaten
    """
    try:
        path = "static/latest_forecast.csv"
        if not os.path.exists(path):
            return jsonify({"error": "Keine Daten verfügbar"}), 404
        
        df = pd.read_csv(path)
        return jsonify({
            "status": "success",
            "data": df.to_dict(orient="records"),
            "meta": {
                "total_bins": len(df),
                "high_urgency": len(df[df['urgency_level'] == 'high']),
                "model_type": "Random Forest",
                "timestamp": datetime.now().isoformat()
            }
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/download")
def download_results():
    """
    Download der Ergebnisse
    """
    try:
        path = "static/latest_forecast.csv"
        if not os.path.exists(path):
            return "Keine Ergebnisse zum Download verfügbar", 404
        
        # Zeitstempel für Dateiname
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"smartbin_rf_forecast_{timestamp}.csv"
        
        return send_file(path, as_attachment=True, download_name=filename)
    except Exception as e:
        logger.error(f"Fehler beim Download: {str(e)}")
        return f"Fehler beim Download: {str(e)}", 500

@app.errorhandler(413)
def too_large(e):
    return render_template("error.html", 
                         error="Datei zu groß. Maximum 16MB erlaubt."), 413

@app.errorhandler(500)
def internal_error(e):
    logger.error(f"Server Fehler: {str(e)}")
    return render_template("error.html", 
                         error="Interner Serverfehler"), 500

if __name__ == "__main__":
    print("🚀 SmartBin Server startet...")
    print("📊 Modell-Typ: Random Forest")
    
    if model_manager.is_available():
        print("✅ Random Forest Modell erfolgreich geladen")
        print(f"📈 Features: {len(model_manager.feature_names)}")
    else:
        print("❌ Random Forest Modell nicht verfügbar - Server wird beendet")
        exit(1)
    
    app.run(debug=True, host="0.0.0.0", port=5000)