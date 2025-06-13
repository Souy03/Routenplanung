import folium
import json
import random

# GeoJSON-Datei laden
with open("muell_nbg.geojson", "r", encoding="utf-8") as f:
    data = json.load(f)

# Karte initialisieren
m = folium.Map(location=[49.44, 11.08], zoom_start=13)

# Farben pro Recycling-Typ
colors = {
    "glass_bottles": "green",
    "clothes": "blue",
    "shoes": "brown",
    "unknown": "gray"
}

container_id = 1  # ID-Zähler

# Marker für jeden Standort setzen
for feat in data["features"]:
    coords = feat["geometry"]["coordinates"]
    tags = feat["properties"]
    lat, lon = coords[1], coords[0]

    # Recycling-Typen extrahieren
    rec_types = [k.replace("recycling:", "") for k in tags if k.startswith("recycling:") and tags[k] == "yes"]

    # Füllstände zufällig erzeugen
    fill_info = ""
    for rec in rec_types:
        fill_level = random.randint(10, 95)  # Füllstand zwischen 10–95%
        fill_info += f"<li><b>{rec.capitalize()}</b>: {fill_level}%</li>"

    if not fill_info:
        fill_info = "<li><i>Keine Daten</i></li>"

    # Popup-Text mit allen Infos
    popup_html = f"""
    <b>Container ID:</b> {container_id}<br>
    <b>Koordinaten:</b> {lat:.6f}, {lon:.6f}<br>
    <b>Füllstand:</b><ul>{fill_info}</ul>
    """

    # Farbe für Haupttyp wählen (1. Typ oder 'unknown')
    main_type = rec_types[0] if rec_types else "unknown"
    color = colors.get(main_type, "gray")

    # Marker setzen
    folium.CircleMarker(
        [lat, lon],
        radius=6,
        color=color,
        fill=True,
        fill_opacity=0.8,
        popup=folium.Popup(popup_html, max_width=300)
    ).add_to(m)

    container_id += 1  # Nächste ID

# HTML exportieren
m.save("nuernberg_recycling_mit_fuellstand.html")
