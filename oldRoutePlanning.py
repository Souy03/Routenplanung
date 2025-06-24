import geopandas as gpd
import folium
import numpy as np
from scipy.spatial.distance import cdist
from ortools.constraint_solver import pywrapcp, routing_enums_pb2

# === 1. GeoJSON laden ===
gdf = gpd.read_file("waste_baskets_nbg.geojson")
gdf["lon"] = gdf.geometry.x
gdf["lat"] = gdf.geometry.y

# === 2. Füllstand zufällig erzeugen (für Tests) ===
np.random.seed(42)
gdf["fuellstand"] = np.random.randint(5, 101, size=len(gdf))

# === 3. Filter: nur volle Container (>70%) ===
gdf = gdf[gdf["fuellstand"] > 70].reset_index(drop=True)

# === 4. Begrenzung auf max. 25 Container für Demo ===
gdf = gdf.iloc[:25].copy()

# === 5. Distanzmatrix berechnen (euklidisch in Meter) ===
coords = gdf[["lat", "lon"]].values
dist_matrix = cdist(coords, coords, metric="euclidean") * 111000

# === 6. OR-Tools Setup ===
data = {
    "distance_matrix": dist_matrix.astype(int).tolist(),
    "num_vehicles": 1,
    "depot": 0,
}

manager = pywrapcp.RoutingIndexManager(len(data["distance_matrix"]), data["num_vehicles"], data["depot"])
routing = pywrapcp.RoutingModel(manager)

def distance_callback(from_index, to_index):
    from_node = manager.IndexToNode(from_index)
    to_node = manager.IndexToNode(to_index)
    return data["distance_matrix"][from_node][to_node]

transit_cb_index = routing.RegisterTransitCallback(distance_callback)
routing.SetArcCostEvaluatorOfAllVehicles(transit_cb_index)

search_params = pywrapcp.DefaultRoutingSearchParameters()
search_params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC

solution = routing.SolveWithParameters(search_params)

# === 7. Route extrahieren ===
route_order = []
if solution:
    index = routing.Start(0)
    while not routing.IsEnd(index):
        route_order.append(manager.IndexToNode(index))
        index = solution.Value(routing.NextVar(index))
    route_order.append(manager.IndexToNode(index))

# === 8. Karte erzeugen ===
m = folium.Map(location=[gdf["lat"].mean(), gdf["lon"].mean()], zoom_start=13)

# Marker setzen
for i, row in gdf.iterrows():
    folium.CircleMarker(
        location=(row["lat"], row["lon"]),
        radius=6,
        color="green",
        fill=True,
        fill_opacity=0.9,
        popup=f"ID: {row['id']}<br>Füllstand: {row['fuellstand']}%"
    ).add_to(m)

# Optimierte Route einzeichnen
route_coords = [(gdf.loc[i, "lat"], gdf.loc[i, "lon"]) for i in route_order]
folium.PolyLine(route_coords, color="red", weight=3).add_to(m)

# Karte speichern
m.save("optimierte_route_nbg.html")
print("✔ Karte gespeichert als optimierte_route_nbg.html")
