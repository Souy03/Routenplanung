# Datei: generate_forecast_map.py

import folium
import json
import pandas as pd
import numpy as np
from shapely.geometry import shape, Point
from forecast import OptimizedSmartBinForecaster
from datetime import datetime

# === Pfade ===
geojson_path = "waste_baskets_with_hotspots.geojson"
polygon_path = "nuernberg_innenstadt_polygon.geojson"
output_html = "nbg_forecast_map.html"
model_path = "models/"

# === Modell initialisieren und laden ===
forecaster = OptimizedSmartBinForecaster(model_path=model_path, use_rules=True)
forecaster.load_model()

# === GeoJSON laden ===
with open(geojson_path, "r", encoding="utf-8") as f:
    geojson_data = json.load(f)

# Innenstadt-Polygon laden
with open(polygon_path, "r", encoding="utf-8") as f:
    polygon_geojson = json.load(f)
    polygon_shape = shape(polygon_geojson["features"][0]["geometry"])

# Eigenschaften setzen und Hotspot-Tags aktualisieren
for idx, feature in enumerate(geojson_data["features"], start=1):
    props = feature.setdefault("properties", {})
    lon, lat = feature["geometry"]["coordinates"]

    props["WasteBasketID"] = props.get("WasteBasketID", idx)
    props["SensorID"] = props.get("SensorID", idx)
    props["fill_level"] = int(np.random.randint(5, 90))  # zufälliger Füllstand
    props["temperature"] = float(np.random.uniform(18, 35))
    props["humidity"] = float(np.random.uniform(50, 95))
    props["timestamp"] = datetime.now().isoformat()

    hotspot_tags = props.get("hotspot_tags", [])
    if "city_center" not in hotspot_tags and polygon_shape.contains(Point(lon, lat)):
        hotspot_tags.append("city_center")
    props["hotspot_tags"] = hotspot_tags

# In DataFrame umwandeln
records = []
for feature in geojson_data["features"]:
    p = feature["properties"]
    lon, lat = feature["geometry"]["coordinates"]
    records.append({
        "WasteBasketID": str(p["WasteBasketID"]),
        "SensorID": str(p["SensorID"]),
        "latitude": lat,
        "longitude": lon,
        "timestamp": p["timestamp"],
        "fill_level": p["fill_level"],
        "temperature": p["temperature"],
        "humidity": p["humidity"],
        "hotspot_tags": p.get("hotspot_tags", [])
    })

raw_df = pd.DataFrame(records)
raw_df["timestamp"] = pd.to_datetime(raw_df["timestamp"])

# Prognose erzeugen
processed_df = forecaster.preprocess_data(raw_df)
forecasts = forecaster.create_forecast(processed_df)

# === Karte vorbereiten ===
m = folium.Map(location=[49.45, 11.08], zoom_start=13)
folium.GeoJson(
    polygon_geojson,
    name="Innenstadt",
    style_function=lambda x: {
        "fillColor": '#4169E1',#"#ffa50055",
        "color":'#4169E1', #"#ffa500",
        "weight": 2.5,
        "fillOpacity": 0.25,
    }
).add_to(m)

# Farbverlauf (diskret, basierend auf Intervallen)
FILL_COLOR_SCALE = [
    (0, 20, "#66BB66"),
    (20, 40, "#99CC66"),
    (40, 60, "#EEEE44"),
    (60, 80, "#FFB266"),
    (80, 101, "#CC6666"),
]

def get_color(fill):
    for min_val, max_val, color in FILL_COLOR_SCALE:
        if min_val <= fill < max_val:
            return color
    return "#888888"

# Marker zeichnen
for forecast in forecasts:
    bin_row = raw_df[raw_df['WasteBasketID'] == forecast['WasteBasketID']].iloc[-1]
    lat, lon = bin_row["latitude"], bin_row["longitude"]
    fill24 = forecast["predicted_fill_24h"]
    color = get_color(fill24)

    popup = f"""
    <b>WasteBasket ID:</b> {forecast['WasteBasketID']}<br>
    <b>Sensor ID:</b> {bin_row['SensorID']}<br>
    <b>Latitude:</b> {lat:.5f}<br>
    <b>Longitude:</b> {lon:.5f}<br><br>
    <b>Aktueller Füllstand:</b> {forecast['current_fill_level']}%<br><br>
    <b>📈 Prognose:</b><br>
    &nbsp;&nbsp;&nbsp;8h:&nbsp;&nbsp; {forecast['predicted_fill_8h']:.1f}%<br>
    &nbsp;&nbsp;&nbsp;16h: {forecast['predicted_fill_16h']:.1f}%<br>
    &nbsp;&nbsp;&nbsp;24h: {forecast['predicted_fill_24h']:.1f}%<br><br>
    <b>Dringlichkeit:</b> {forecast['urgency_level'].capitalize()}<br>
    <b>Regel angewendet:</b> {forecast.get('rule_applied', 'none')}<br>
    """

    folium.CircleMarker(
        location=[lat, lon],
        radius=6,
        color="#AA00FF" if bin_row["hotspot_tags"] else "#1874cd",
        weight=2,
        fill=True,
        fill_color=color,
        fill_opacity=0.9,
        popup=folium.Popup(popup, max_width=300)
    ).add_to(m)

# Legende mit Ein-/Ausklappfunktion
legend_html = """
<style>
  #legend-toggle-btn {
    position: fixed;
    bottom: 30px;
    left: 30px;
    z-index: 9999;
    background: white;
    border: 2px solid grey;
    border-radius: 6px;
    padding: 4px 8px;
    font-size: 16px;
    cursor: pointer;
    box-shadow: 1px 1px 4px rgba(0,0,0,0.3);
  }
  #map-legend {
    position: fixed;
    bottom: 30px;
    left: 70px;
    width: 190px;
    z-index: 9998;
    background-color: white;
    border:2px solid grey;
    border-radius:6px;
    padding: 10px;
    font-size:14px;
    box-shadow: 2px 2px 6px rgba(0,0,0,0.3);
    display: none;
  }
</style>

<div id="legend-toggle-btn" onclick="toggleLegend()">ℹ️</div>

<div id="map-legend">
  <b>Füllstandsprognose (%)</b>
  <ul style="list-style:none; padding-left:0; margin:0;">
    <li><span style="background:#66BB66;width:12px;height:12px;display:inline-block;margin-right:6px;border:1px solid #1874cd"></span>0–19</li>
    <li><span style="background:#99CC66;width:12px;height:12px;display:inline-block;margin-right:6px;border:1px solid #1874cd"></span>20–39</li>
    <li><span style="background:#EEEE44;width:12px;height:12px;display:inline-block;margin-right:6px;border:1px solid #1874cd"></span>40–59</li>
    <li><span style="background:#FFB266;width:12px;height:12px;display:inline-block;margin-right:6px;border:1px solid #1874cd"></span>60–79</li>
    <li><span style="background:#CC6666;width:12px;height:12px;display:inline-block;margin-right:6px;border:1px solid #1874cd"></span>80–100</li>
  </ul>
</div>

<script>
function toggleLegend() {
  var legend = document.getElementById('map-legend');
  if (legend.style.display === 'none') {
    legend.style.display = 'block';
  } else {
    legend.style.display = 'none';
  }
}
</script>
"""

# Legende zur Karte hinzufügen
m.get_root().html.add_child(folium.Element(legend_html))


# Karte speichern
m.save(output_html)
print(f"✅ Prognosekarte gespeichert: {output_html}")
