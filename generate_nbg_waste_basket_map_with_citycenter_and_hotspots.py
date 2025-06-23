import folium
import json
import random
from shapely.geometry import shape, Point

# Datei-Pfade
geojson_path = "waste_baskets_with_hotspots.geojson"
polygon_path = "nuernberg_innenstadt_polygon.geojson"
output_html = "nbg_wastebaskets_with_hotspots_and_citycenter_map.html"

# Farbverlauf nach Füllstand – angepasst für bessere Lesbarkeit mit Hotspot-Farbe
FILL_COLOR_SCALE = [            # Farbverlauf je nach Füllstand
    (0, 20, "#66BB66"),        # Grün
    (20, 40, "#99CC66"),       # Gelbgrün
    (40, 60, "#EEEE44"),       # Gelb
    (60, 80, "#FFB266"),       # Orange
    (80, 101, "#CC6666"),      # Rot
]

# Hotspot-Markerrahmenfarbe (lila)
HOTSPOT_BORDER = "#AA00FF"
# Standardrahmenfarbe (blau)
DEFAULT_BORDER = "#1874cd"

# Karte initialisieren
m = folium.Map(location=[49.45, 11.08], zoom_start=13)

# Innenstadt-Polygon laden und anzeigen
with open(polygon_path, "r", encoding="utf-8") as f:
    polygon_geojson = json.load(f)
    polygon_shape = shape(polygon_geojson["features"][0]["geometry"])

folium.GeoJson(
    polygon_geojson,
    name="Innenstadt",
    style_function=lambda x: {
        "fillColor": "#ffa50055",  # transparenter Orangeton
        "color": "#ffa500",  # Randfarbe
        "weight": 2.5,
        "fillOpacity": 0.25,
    }
).add_to(m)

# GeoJSON mit Mülleimern laden
with open(geojson_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# IDs zurücksetzen und iterieren
for idx, feature in enumerate(data["features"], start=1):
    lon, lat = feature["geometry"]["coordinates"]
    props = feature.setdefault("properties", {})

    # Laufende IDs vergeben
    props["WasteBasketID"] = props.get("WasteBasketID", idx)
    props["SensorID"] = props.get("SensorID", idx)

    # Hotspot-Tags (falls vorhanden)
    hotspot_tags = props.get("hotspot_tags", [])

    # Falls in Polygon, aber kein "city_center" markiert → ergänzen
    if "city_center" not in hotspot_tags and polygon_shape.contains(Point(lon, lat)):
        hotspot_tags.append("city_center")

    # Füllstand zufällig erzeugen
    fill_level = random.randint(5, 100)
    props["fill_level"] = fill_level

    # Füllfarbe berechnen
    fill_color = next(
        (color for min_val, max_val, color in FILL_COLOR_SCALE if min_val <= fill_level < max_val),
        "#888888"
    )

    # Rahmenfarbe bestimmen
    border_color = HOTSPOT_BORDER if hotspot_tags else DEFAULT_BORDER

    # Popup-Text
    popup = f"""
    <b>WasteBasket ID:</b> {props['WasteBasketID']}<br>
    <b>Sensor ID:</b> {props['SensorID']}<br>
    <b>Latitude:</b> {lat:.6f}<br>
    <b>Longitude:</b> {lon:.6f}<br>
    <b>Füllstand:</b> {fill_level}%<br>
    <b>Hotspot-Tags:</b> {", ".join(hotspot_tags) if hotspot_tags else "–"}
    """

    # Marker hinzufügen
    folium.CircleMarker(
        location=[lat, lon],
        radius=6,
        color=border_color,
        weight=2,
        fill=True,
        fill_color=fill_color,
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
    width: 180px;
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
  <b>Füllstand (%)</b>
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
print(f"✅ Karte gespeichert: {output_html}")

