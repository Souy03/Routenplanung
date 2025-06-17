import folium
import json
import random

# GeoJSON-Datei mit Mülleimern laden
with open("waste_baskets_nbg.geojson", "r", encoding="utf-8") as f:
    data = json.load(f)

# Füllstandsabhängige Farbskala (Obergrenzen exklusiv)
def get_color(fill):
    if fill < 20:
        return "#66BB66"  # Grün
    elif fill < 40:
        return "#99CC66"  # Gelbgrün
    elif fill < 60:
        return "#EEEE44"  # Gelb
    elif fill < 80:
        return "#FFB266"  # Orange
    else:
        return "#CC6666"  # Rot

# Karte initialisieren
m = folium.Map(location=[49.45, 11.08], zoom_start=13)

# Mülleimer-Marker mit zufälligem Füllstand zeichnen
waste_id = 1
for feature in data["features"]:
    lon, lat = feature["geometry"]["coordinates"]
    fill_level = random.randint(5, 100)

    popup = f"""
    <b>WasteBasket ID:</b> {waste_id}<br>
    <b>Latitude:</b> {lat:.6f}<br>
    <b>Longitude:</b> {lon:.6f}<br>
    <b>Füllstand:</b> {fill_level}%
    """

    folium.CircleMarker(
        location=[lat, lon],
        radius=6,
        color="#1874cd",         # Blauer Rahmen
        weight=1,
        fill=True,
        fill_color=get_color(fill_level),
        fill_opacity=0.9,
        popup=folium.Popup(popup, max_width=250)
    ).add_to(m)

    waste_id += 1

# HTML + JS für Legende & Toggle-Symbol
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
m.save("nbg_wastebaskets_map.html")
