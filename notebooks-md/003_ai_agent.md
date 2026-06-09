```python
# =============================================================================
# NOTEBOOK 4 — AI AGENT (GOAL-BASED)
# Earthquake Disaster Response System
#
# Agent lifecycle for one earthquake event:
#   PERCEPT  → load earthquake input (lat, lon, mag, depth, etc.)
#   PERCEIVE → run tsunami classifier + threat classifier
#   GOAL     → find nearest city to the epicentre  (the place that needs help)
#   PLAN     → find nearest hub airport to goal city
#              build airport graph (hub + nearby airports + goal airport)
#              run A* to find cheapest route, land/water cost penalises
#              crossings between the two mediums
#   ACT      → output route, total cost, dispatch order
#   VISUALISE→ Folium map with full story
# =============================================================================
```


```python
import numpy as np
import pandas as pd
import folium
import joblib
import heapq
import os
import warnings
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from math       import radians, sin, cos, sqrt, atan2
from global_land_mask import globe
warnings.filterwarnings('ignore')
```

## Load All Saved Artefacts


```python
MODEL_DIR = '../models/'
 
tsunami_model    = joblib.load(MODEL_DIR + 'tsunami_model.pkl')
tsunami_scaler   = joblib.load(MODEL_DIR + 'tsunami_scaler.pkl')
tsunami_features = joblib.load(MODEL_DIR + 'tsunami_features.pkl')
 
threat_model    = joblib.load(MODEL_DIR + 'threat_model.pkl')
threat_scaler   = joblib.load(MODEL_DIR + 'threat_scaler.pkl')
threat_le       = joblib.load(MODEL_DIR + 'threat_label_encoder.pkl')
threat_features = joblib.load(MODEL_DIR + 'threat_features.pkl')
 
hubs_df = pd.read_csv(MODEL_DIR + 'supply_hubs.csv')
 
print("All models and hubs loaded.")
print(f"Supply hubs:\n{hubs_df[['hub_id','airport_name','iata_code','hub_lat','hub_lon']]}")
```

    All models and hubs loaded.
    Supply hubs:
       hub_id                                       airport_name iata_code  \
    0       0                                     Handan Airport       HDG   
    1       1                        Mopah International Airport       MKQ   
    2       2                     Mataveri International Airport       IPC   
    3       3                                   Sandspit Airport       YZP   
    4       4  Augusto C. Sandino (Managua) International Air...       MGA   
    
         hub_lat     hub_lon  
    0  36.524824  114.424126  
    1  -8.523898  140.419693  
    2 -27.165411 -109.421027  
    3  53.254299 -131.813995  
    4  12.141500  -86.168198  


## Load Supporting Datasets


```python
apt = pd.read_csv('../data/airports.csv')
cit = pd.read_csv('../data/worldcities.csv')          # adjust path
 
# Clean airports
apt_clean = (
    apt[apt['type'].isin(['large_airport','medium_airport'])]
      .dropna(subset=['latitude_deg','longitude_deg'])
      .rename(columns={'latitude_deg':'lat','longitude_deg':'lon'})
      [['name','iata_code','type','lat','lon','iso_country','municipality']]
      .reset_index(drop=True)
)
 
# Clean cities — need lat, lng, population
cit_clean = (
    cit.dropna(subset=['lat','lng','population'])
       .copy()
)
cit_clean['lat'] = pd.to_numeric(cit_clean['lat'], errors='coerce')
cit_clean['lng'] = pd.to_numeric(cit_clean['lng'], errors='coerce')
cit_clean['population'] = pd.to_numeric(cit_clean['population'], errors='coerce')
cit_clean.dropna(subset=['lat','lng'], inplace=True)
 
print(f"Airports: {apt_clean.shape[0]}  |  Cities: {cit_clean.shape[0]}")
 
```

    Airports: 5277  |  Cities: 48778


## Utility Functions


```python
def haversine_km(lat1, lon1, lat2, lon2):
    """Distance in km between one point and an array of points."""
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians,
                                  [np.atleast_1d(lat1), np.atleast_1d(lon1),
                                   np.atleast_1d(lat2), np.atleast_1d(lon2)])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return np.squeeze(2 * R * np.arcsin(np.sqrt(a)))
 
 
def is_land(lat, lon):
    """True if the coordinate is on land (uses global_land_mask)."""
    return bool(globe.is_land(lat, lon))
 
 
def edge_cost(lat1, lon1, lat2, lon2):
    """
    Travel cost between two airport nodes.
    Base cost = distance in km.
    Land-to-water or water-to-land transition adds a 3× penalty
    (reflects the modal shift: truck → ship or ship → truck).
    Pure over-water segments get a 2× penalty (slower than air).
    Pure over-land segments: no penalty.
 
    In practice, since all nodes are airports (on land), the penalty
    fires when the great-circle path between two airports passes through
    a midpoint that is over water — a simple proxy for a sea crossing.
    """
    dist = haversine_km(lat1, lon1, lat2, lon2)
 
    src_land  = is_land(lat1, lon1)
    dst_land  = is_land(lat2, lon2)
    mid_lat   = (lat1 + lat2) / 2
    mid_lon   = (lon1 + lon2) / 2
    mid_land  = is_land(mid_lat, mid_lon)
 
    if src_land and dst_land and mid_land:
        multiplier = 1.0    # pure overland — cheap
    elif not mid_land:
        multiplier = 2.0    # path crosses water
    else:
        multiplier = 1.5    # coastal hop
 
    return dist * multiplier
 
 
def nearest_airports(lat, lon, apt_df, k=40):
    """
    Returns the k nearest airports to a given coordinate, sorted by distance.
    """
    # Calculate distances to all airports using your existing haversine function
    dists = haversine_km(lat, lon, apt_df['lat'].values, apt_df['lon'].values)
    
    # Get the indices of the smallest distances
    idx = np.argsort(dists)[:k]
    
    # Extract those rows and add the distance column
    result = apt_df.iloc[idx].copy()
    result['dist_km'] = dists[idx]
    
    return result.reset_index(drop=True)
 
 
def nearest_city(lat, lon, city_df, top_n=5):
    """
    Return the nearest populated city to the epicentre.
    Prefers cities with population > 10,000 (actual affected population,
    not a village that may have 50 people).
    """
    populated = city_df[city_df['population'] >= 10_000].copy()
    dists = haversine_km(lat, lon,
                         populated['lat'].values,
                         populated['lng'].values)
    populated = populated.copy()
    populated['dist_km'] = dists
    return populated.nsmallest(top_n, 'dist_km').iloc[0]
 
 
def nearest_hub(lat, lon, hubs):
    """Return the hub row nearest to (lat, lon)."""
    dists = haversine_km(lat, lon,
                         hubs['hub_lat'].values,
                         hubs['hub_lon'].values)
    return hubs.iloc[np.argmin(dists)].copy()
```

## A* Pathfinder


```python
# Graph representation:
#   Nodes  → integer indices into a combined node list
#             (hub airport + intermediate airports + goal airport)
#   Edges  → all pairs within MAX_EDGE_KM km of each other
#   Cost   → edge_cost() — distance × land/water multiplier
#   Heuristic → straight-line haversine to goal (admissible)
#
# We limit the graph to airports within SEARCH_RADIUS_KM of the
# earthquake to keep the graph tractable and geographically relevant.
 
MAX_EDGE_KM      = 2500   # max direct flight distance (km)
SEARCH_RADIUS_KM = 6000   # how far out we pull candidate airports
INTERMEDIATE_N   = 60     # how many airports to include in the graph
 
 
def build_graph(hub_row, goal_apt_row, apt_df):
    hub_lat, hub_lon = hub_row['hub_lat'], hub_row['hub_lon']
    goal_lat, goal_lon = goal_apt_row['lat'], goal_apt_row['lon']

    # 1. Pull airports at endpoints
    near_hub  = nearest_airports(hub_lat, hub_lon, apt_df, k=40)
    near_goal = nearest_airports(goal_lat, goal_lon, apt_df, k=30)
    
    # 2. Pull "stepping stone" airports at the midpoint to bridge oceans
    mid_lat, mid_lon = (hub_lat + goal_lat) / 2, (hub_lon + goal_lon) / 2
    stepping_stones  = nearest_airports(mid_lat, mid_lon, apt_df, k=20)

    candidates = pd.concat([near_hub, stepping_stones, near_goal], ignore_index=True)

    # 3. Build unique node list safely
    hub_node  = {'name': hub_row['airport_name'], 'iata': str(hub_row.get('iata_code', '---')), 'lat': hub_lat, 'lon': hub_lon}
    goal_node = {'name': goal_apt_row['name'], 'iata': str(goal_apt_row.get('iata_code', '---')), 'lat': goal_lat, 'lon': goal_lon}

    nodes = [hub_node]
    seen  = {(round(hub_lat, 2), round(hub_lon, 2))}

    for _, r in candidates.iterrows():
        key = (round(r['lat'], 2), round(r['lon'], 2))
        if key not in seen:
            seen.add(key)
            nodes.append({'name': r['name'], 'iata': str(r['iata_code']), 'lat': r['lat'], 'lon': r['lon']})

    # Ensure the goal is appended if it wasn't already caught
    goal_key = (round(goal_lat, 2), round(goal_lon, 2))
    if goal_key not in seen:
        nodes.append(goal_node)

    # 4. Safely locate exact indices
    start_idx = 0
    goal_idx  = next((i for i, n in enumerate(nodes) if n['name'] == goal_node['name']), len(nodes) - 1)

    # 5. Build Adjacency list
    adj = {i: [] for i in range(len(nodes))}
    for i in range(len(nodes)):
        for j in range(len(nodes)):
            if i == j: continue
            
            d = float(np.squeeze(haversine_km(nodes[i]['lat'], nodes[i]['lon'], [nodes[j]['lat']], [nodes[j]['lon']])))
            if d <= MAX_EDGE_KM:
                cost = edge_cost(nodes[i]['lat'], nodes[i]['lon'], nodes[j]['lat'], nodes[j]['lon'])
                adj[i].append((j, cost))

    return nodes, adj, start_idx, goal_idx
 
 
def astar(nodes, adj, start, goal):
    """
    A* search on the airport graph.
    Returns (path as list of node indices, total cost) or (None, inf).
    """
    def h(node_idx):
        return haversine_km(nodes[node_idx]['lat'], nodes[node_idx]['lon'],
                            nodes[goal]['lat'],     nodes[goal]['lon'])
 
    open_set = [(h(start), 0.0, start, [start])]   # (f, g, node, path)
    visited  = {}
 
    while open_set:
        f, g, current, path = heapq.heappop(open_set)
 
        if current == goal:
            return path, g
 
        if current in visited and visited[current] <= g:
            continue
        visited[current] = g
 
        for neighbour, cost in adj[current]:
            new_g = g + cost
            new_f = new_g + h(neighbour)
            heapq.heappush(open_set, (new_f, new_g, neighbour, path + [neighbour]))
 
    return None, float('inf')
```

## ML Inference Helpers


```python
def build_model_row(event, features, extra_cols=None):
    """
    Build a one-row DataFrame matching the model's expected feature list.
    event: dict with earthquake properties.
    extra_cols: additional zero-filled columns (e.g. one-hot magType dummies).
    """
    row = {f: event.get(f, 0) for f in features}
    df_row = pd.DataFrame([row])
    # Fill any missing dummy columns with 0
    for col in features:
        if col not in df_row.columns:
            df_row[col] = 0
    return df_row[features]
 
 
def predict_tsunami(event):
    row    = build_model_row(event, tsunami_features)
    scaled = tsunami_scaler.transform(row)
    pred   = tsunami_model.predict(scaled)[0]
    prob   = tsunami_model.predict_proba(scaled)[0][1] if hasattr(tsunami_model, 'predict_proba') else None
    return int(pred), prob
 
 
def predict_threat(event):
    row    = build_model_row(event, threat_features)
    scaled = threat_scaler.transform(row)
    pred   = threat_model.predict(scaled)[0]
    label  = threat_le.inverse_transform([pred])[0]
    return label
 
def nearest_hub(goal_lat, goal_lon, hubs):
    """
    Finds the closest supply hub to the TARGET CITY, not the epicentre.
    """
    dists = haversine_km(goal_lat, goal_lon, 
                         hubs['hub_lat'].values, hubs['hub_lon'].values)
    
    # Returns the single closest hub row
    return hubs.iloc[np.argmin(dists)].copy()
```

## The Agent Class


```python
class DisasterResponseAgent:
    """
    Goal-based AI agent for earthquake disaster response.
 
    AGENT CYCLE (one earthquake event):
      1. PERCEPT     — receive raw earthquake reading
      2. PERCEIVE    — run ML models to assess tsunami risk and threat level
      3. GOAL        — identify the affected city (nearest populated place)
      4. PLAN        — find supply hub → build graph → run A*
      5. ACT         — output dispatch order
      6. DISPLAY     — produce Folium map
    """
 
    THREAT_COLORS = {
        'LOW'     : '#2ecc71',
        'MEDIUM'  : '#f1c40f',
        'HIGH'    : '#e67e22',
        'CRITICAL': '#e74c3c',
    }
 
    THREAT_SUPPLIES_LAND = {
        'LOW'     : '2 trucks (basic supplies, 20t)',
        'MEDIUM'  : '5 trucks + 1 medical team (50t)',
        'HIGH'    : '10 trucks + 3 medical teams + 1 field hospital (150t)',
        'CRITICAL': '30 trucks + 10 medical teams + 2 field hospitals (500t)',
    }
    THREAT_SUPPLIES_AIR = {
        'LOW'     : '2 cargo flights (basic supplies, 20t)',
        'MEDIUM'  : '4 cargo flights + 1 medical team (50t)',
        'HIGH'    : '8 cargo flights + 3 medical teams + field hospital (150t)',
        'CRITICAL': '20 cargo flights + 10 medical teams + 2 field hospitals (500t)',
    }
    THREAT_SUPPLIES_MARITIME = {
        'LOW'     : '1 naval vessel (basic supplies)',
        'MEDIUM'  : '2 naval vessels + medical team',
        'HIGH'    : '4 naval vessels + hospital ship + rescue teams',
        'CRITICAL': 'Full naval task force + hospital ship + 10 rescue teams',
    }
 
    def __init__(self, hubs_df, apt_df, city_df):
        self.hubs    = hubs_df
        self.airports = apt_df
        self.cities   = city_df
 
    def _classify_dispatch_mode(self, path, goal_city):
        """
        Determine dispatch mode:
          'maritime'  — A* failed entirely (isolated island, no airport chain)
          'air'       — default for multi-airport hops across our graph
        """
        if path is None or len(path) == 0:
            return 'maritime'
            
        # In an airport-to-airport graph, hops are flown.
        # (You could add logic here to return 'land' if distance < 500km and on same landmass)
        return 'air'
 
    # ── Step 1+2: Percept + Perceive ──────────────────────────────────────────
    def perceive(self, event):
        # 1. Get Base ML predictions
        ml_pred, ml_prob = predict_tsunami(event)
        
        # 2. Apply the physics safety override (FIX 2)
        tsunami_pred, tsunami_prob = self.physics_tsunami_override(event, ml_pred, ml_prob)
        
        # 3. Get threat level
        threat_label = predict_threat(event)
        
        return {
            'tsunami_prediction' : tsunami_pred,
            'tsunami_probability': round(tsunami_prob, 3) if tsunami_prob else None,
            'threat_level'       : threat_label,
        }
 
    # ── Step 3: Goal ──────────────────────────────────────────────────────────
    def formulate_goal(self, event):
        city = nearest_city(event['latitude'], event['longitude'], self.cities)
        return city
 
    # ── Step 4: Plan ──────────────────────────────────────────────────────────
    def plan_route(self, event, goal_city):
        # FIX 3: Pass goal_city coordinates instead of event coordinates
        hub_row = nearest_hub(goal_city['lat'], goal_city['lng'], self.hubs)
        
        # Ensure nearest_airport returns a single row correctly
        goal_airport_df = nearest_airport(goal_city['lat'], goal_city['lng'], self.airports, k=1)
        goal_airport = goal_airport_df.iloc[0]
        
        nodes, adj, start_idx, goal_idx = build_graph(hub_row, goal_airport, self.airports)
        path, cost = astar(nodes, adj, start_idx, goal_idx)
        
        return hub_row, goal_airport, goal_city, nodes, path, cost
 
    def physics_tsunami_override(self, event, ml_pred, ml_prob):
        """
        Safety layer: Overrides ML prediction to YES if physics dictate a tsunami is certain.
        """
        mag   = event.get('magnitude', 0)
        depth = event.get('depth', 999)
        lat   = event.get('latitude', 0)
        lon   = event.get('longitude', 0)
        
        # Check if epicentre is over water
        over_water = not is_land(lat, lon)
        
        # Rule 1: M8.5+, shallow, over water
        if (mag >= 8.5) and (depth <= 70) and over_water and ml_pred == 0:
            print(f"  [PHYSICS OVERRIDE] M{mag} shallow ({depth}km) over water — overriding to YES.")
            return 1, 1.0
            
        # Rule 2: M9.0+ anywhere (massive energy displacement)
        if mag >= 9.0 and ml_pred == 0:
            print(f"  [PHYSICS OVERRIDE] M{mag} >= 9.0 — overriding tsunami to YES.")
            return 1, 1.0
            
        return ml_pred, ml_prob
 
    # ── Step 5: Act ───────────────────────────────────────────────────────────
    def act(self, perception, hub_row, goal_city, goal_airport, nodes, path, cost):
        threat  = perception['threat_level']
        t_pred  = perception['tsunami_prediction']
        t_prob  = perception['tsunami_probability']

        # 1. Figure out the transportation mode
        mode = self._classify_dispatch_mode(path, goal_city)

        # 2. Pick the correct dictionary and label based on the mode
        if mode == 'maritime':
            supplies = self.THREAT_SUPPLIES_MARITIME[threat]
            mode_str = 'MARITIME DISPATCH (no air bridge available)'
        elif mode == 'air':
            supplies = self.THREAT_SUPPLIES_AIR[threat]
            mode_str = 'AIR DISPATCH (cargo / charter flights)'
        else:
            supplies = self.THREAT_SUPPLIES_LAND[threat]
            mode_str = 'LAND DISPATCH (trucks / ground convoy)'

        print("\n" + "="*60)
        print("  DISASTER RESPONSE AGENT — DISPATCH ORDER")
        print("="*60)
        print(f"  Threat level   : {threat}")
        print(f"  Tsunami risk   : {'YES' if t_pred else 'NO'}"
              + (f"  (prob={t_prob:.1%})" if t_prob is not None else ""))
        print(f"  Target city    : {goal_city['city']} ({goal_city['country']})")
        print(f"  Nearest airport: {goal_airport['name']}")
        print(f"  Supply hub     : {hub_row['airport_name']} [{hub_row['iata_code']}]")
        print(f"  Dispatch mode  : {mode_str}")  # Shows the mode in the terminal
        
        if path:
            print(f"  Route hops     : {len(path)} airports")
            print(f"  Route cost     : {cost:,.0f} km-equiv")
        else:
            print(f"  Route hops     : 0 (Unreachable)")
            print(f"  Route cost     : Infinite (No valid flight path)")
            
        # 3. Print the dynamically selected supplies (Trucks, Planes, or Ships)
        print(f"\n  Supplies dispatched: {supplies}")
        if t_pred:
            print("  + Coastal rescue units + tsunami relief kits")
        print("="*60)

        if path:
            print("\n  ROUTE:")
            for i, node_idx in enumerate(path):
                n = nodes[node_idx]
                iata = n['iata'] if pd.notna(n['iata']) else '---'
                print(f"    {'START →' if i==0 else ('→ END  ' if i==len(path)-1 else '  →    ')} "
                      f"[{iata}] {n['name']}")
                
        return mode
 
    # ── Step 6: Display ───────────────────────────────────────────────────────
    def display_map(self, event, perception, hub_row, goal_city, goal_airport,
                    nodes, path, cost, save_path='../maps/agent_response_map.html'):
 
        threat       = perception['threat_level']
        threat_color = self.THREAT_COLORS[threat]
        map_center   = [(event['latitude'] + goal_city['lat']) / 2,
                        (event['longitude'] + goal_city['lng']) / 2]
 
        m = folium.Map(location=map_center, zoom_start=4,
                       tiles='CartoDB positron')
 
        # ── Earthquake epicentre
        folium.CircleMarker(
            location=[event['latitude'], event['longitude']],
            radius=18, color=threat_color, fill=True,
            fill_opacity=0.35, weight=2.5,
            popup=folium.Popup(
                f"<b>Earthquake epicentre</b><br>"
                f"Magnitude: {event.get('magnitude','?')}<br>"
                f"Depth: {event.get('depth','?')} km<br>"
                f"Threat: <b>{threat}</b><br>"
                f"Tsunami: {'⚠ YES' if perception['tsunami_prediction'] else 'No'}",
                max_width=250
            ),
            tooltip=f"Epicentre — {threat} threat"
        ).add_to(m)
        folium.Marker(
            location=[event['latitude'], event['longitude']],
            icon=folium.DivIcon(
                html=f'<div style="font-size:20px;">🔴</div>',
                icon_size=(30, 30), icon_anchor=(15, 15)
            )
        ).add_to(m)
 
        # ── All supply hubs
        for _, hub in self.hubs.iterrows():
            is_active = hub['hub_id'] == hub_row['hub_id']
            folium.Marker(
                location=[hub['hub_lat'], hub['hub_lon']],
                icon=folium.Icon(
                    color='red' if is_active else 'gray',
                    icon='plane', prefix='fa'
                ),
                popup=folium.Popup(
                    f"<b>{'★ ACTIVE ' if is_active else ''}Supply Hub {hub['hub_id']}</b><br>"
                    f"{hub['airport_name']}<br>"
                    f"IATA: {hub['iata_code']}",
                    max_width=220
                ),
                tooltip=f"{'★ Active hub' if is_active else 'Hub'}: {hub['iata_code']}"
            ).add_to(m)
 
        # ── Target city
        folium.Marker(
            location=[goal_city['lat'], goal_city['lng']],
            icon=folium.Icon(color='blue', icon='home', prefix='fa'),
            popup=folium.Popup(
                f"<b>Target city: {goal_city['city']}</b><br>"
                f"Country: {goal_city['country']}<br>"
                f"Population: {int(goal_city['population']):,}<br>"
                f"Nearest airport: {goal_airport['name']}",
                max_width=250
            ),
            tooltip=f"Target: {goal_city['city']}"
        ).add_to(m)
 
        # ── Route line (A* path)
        if path:
            route_coords = [[nodes[i]['lat'], nodes[i]['lon']] for i in path]
            folium.PolyLine(
                locations=route_coords,
                color=threat_color, weight=4, opacity=0.85,
                dash_array=None,
                tooltip=f"A* route — {cost:,.0f} km-equiv"
            ).add_to(m)
 
            # Intermediate waypoints (not hub, not goal)
            for node_idx in path[1:-1]:
                n = nodes[node_idx]
                folium.CircleMarker(
                    location=[n['lat'], n['lon']],
                    radius=5, color='white', fill=True,
                    fill_color=threat_color, fill_opacity=0.9, weight=1.5,
                    tooltip=f"Via: {n.get('iata','?')} — {n['name']}"
                ).add_to(m)
 
        # ── Legend (HTML overlay)
        legend_html = f"""
        <div style="position:fixed;bottom:30px;left:30px;z-index:9999;
                    background:white;padding:14px 18px;border-radius:8px;
                    box-shadow:0 2px 8px rgba(0,0,0,.2);font-size:13px;
                    border-left:5px solid {threat_color}">
          <b>Disaster Response Agent</b><br><br>
          Threat level: <b style="color:{threat_color}">{threat}</b><br>
          Tsunami risk: <b>{'⚠ YES' if perception['tsunami_prediction'] else 'No'}</b><br>
          Target city: <b>{goal_city['city']}</b><br>
          Active hub: <b>{hub_row['iata_code']}</b><br>
          Route hops: <b>{len(path)}</b><br>
          Route cost: <b>{cost:,.0f} km-equiv</b><br><br>
          <span style="color:{threat_color}">●</span> Epicentre &nbsp;
          ✈ Hub &nbsp; 🏠 Target &nbsp;
          <span style="color:{threat_color}">—</span> Route
        </div>
        """
        m.get_root().html.add_child(folium.Element(legend_html))
 
        m.save(save_path)
        print(f"\nMap saved → {save_path}")
        return m
 
    # ── Full pipeline ─────────────────────────────────────────────────────────
    def respond(self, event, save_map=True):
        print(f"\nAgent received event: lat={event['latitude']:.3f}, "
              f"lon={event['longitude']:.3f}, mag={event.get('magnitude','?')}")
 
        perception  = self.perceive(event)
        goal_city   = self.formulate_goal(event)
        hub_row, goal_airport, goal_city, nodes, path, cost = self.plan_route(event, goal_city)
        self.act(perception, hub_row, goal_city, goal_airport, nodes, path, cost)
 
        if save_map:
            m = self.display_map(event, perception, hub_row, goal_city,
                                 goal_airport, nodes, path, cost)
            return m
 
        return perception, hub_row, goal_city, goal_airport, nodes, path, cost
 
```

## Instantiate Agent


```python
agent = DisasterResponseAgent(
    hubs_df   = hubs_df,
    apt_df    = apt_clean,
    city_df   = cit_clean
)
print("Agent ready.")
```

    Agent ready.


## Run Agent on a Sample Earthquake


```python
# We use a real event from the dataset — the M7.0 Solomon Islands earthquake.
# The feature values must match those that the models were trained on.
# Zero-fill any one-hot dummy columns not present (they'll be 0 by default).
 
sample_event = {
    # Core seismic features
    'latitude'   : -9.7963,
    'longitude'  : 159.596,
    'magnitude'  : 7.0,
    'depth'      : 14.0,
    'cdi'        : 8,
    'mmi'        : 7,
    'sig'        : 768,
    'nst'        : 117,
    'dmin'       : 0.509,
    'gap'        : 17.0,
    # Temporal (from date 22-11-2022)
    'month_sin'  : np.sin(2 * np.pi * 11 / 12),
    'month_cos'  : np.cos(2 * np.pi * 11 / 12),
    'year'       : 2022,
    # magType one-hot — this event uses 'mww'
    # The dummy column name depends on how pd.get_dummies named it in training.
    # Common format: 'magType_mww' — check tsunami_features list to confirm.
    'magType_mww': 1,
}
 
map_output = agent.respond(sample_event, save_map=True)
map_output   # renders inline
 
```

    
    Agent received event: lat=-9.796, lon=159.596, mag=7.0
    
    ============================================================
      DISASTER RESPONSE AGENT — DISPATCH ORDER
    ============================================================
      Threat level   : MEDIUM
      Tsunami risk   : YES  (prob=100.0%)
      Target city    : Honiara (Solomon Islands)
      Nearest airport: Honiara International Airport
      Supply hub     : Mopah International Airport [MKQ]
      Dispatch mode  : AIR DISPATCH (cargo / charter flights)
      Route hops     : 4 airports
      Route cost     : 2,521 km-equiv
    
      Supplies dispatched: 4 cargo flights + 1 medical team (50t)
      + Coastal rescue units + tsunami relief kits
    ============================================================
    
      ROUTE:
        START → [MKQ] Mopah International Airport
          →     [OPU] Balimo Airport
          →     [HID] Horn Island Airport
        → END   [HIR] Honiara International Airport
    
    Map saved → ../maps/agent_response_map.html





<div style="width:100%;"><div style="position:relative;width:100%;height:0;padding-bottom:60%;"><span style="color:#565656">Make this Notebook Trusted to load map: File -> Trust Notebook</span><iframe srcdoc="&lt;!DOCTYPE html&gt;
&lt;html&gt;
&lt;head&gt;

    &lt;meta http-equiv=&quot;content-type&quot; content=&quot;text/html; charset=UTF-8&quot; /&gt;
    &lt;script src=&quot;https://cdn.jsdelivr.net/npm/leaflet@1.9.3/dist/leaflet.js&quot;&gt;&lt;/script&gt;
    &lt;script src=&quot;https://code.jquery.com/jquery-3.7.1.min.js&quot;&gt;&lt;/script&gt;
    &lt;script src=&quot;https://cdn.jsdelivr.net/npm/bootstrap@5.2.2/dist/js/bootstrap.bundle.min.js&quot;&gt;&lt;/script&gt;
    &lt;script src=&quot;https://cdnjs.cloudflare.com/ajax/libs/Leaflet.awesome-markers/2.0.2/leaflet.awesome-markers.js&quot;&gt;&lt;/script&gt;
    &lt;link rel=&quot;stylesheet&quot; href=&quot;https://cdn.jsdelivr.net/npm/leaflet@1.9.3/dist/leaflet.css&quot;/&gt;
    &lt;link rel=&quot;stylesheet&quot; href=&quot;https://cdn.jsdelivr.net/npm/bootstrap@5.2.2/dist/css/bootstrap.min.css&quot;/&gt;
    &lt;link rel=&quot;stylesheet&quot; href=&quot;https://netdna.bootstrapcdn.com/bootstrap/3.0.0/css/bootstrap-glyphicons.css&quot;/&gt;
    &lt;link rel=&quot;stylesheet&quot; href=&quot;https://cdn.jsdelivr.net/npm/@fortawesome/fontawesome-free@6.2.0/css/all.min.css&quot;/&gt;
    &lt;link rel=&quot;stylesheet&quot; href=&quot;https://cdnjs.cloudflare.com/ajax/libs/Leaflet.awesome-markers/2.0.2/leaflet.awesome-markers.css&quot;/&gt;
    &lt;link rel=&quot;stylesheet&quot; href=&quot;https://cdn.jsdelivr.net/gh/python-visualization/folium/folium/templates/leaflet.awesome.rotate.min.css&quot;/&gt;

            &lt;meta name=&quot;viewport&quot; content=&quot;width=device-width,
                initial-scale=1.0, maximum-scale=1.0, user-scalable=no&quot; /&gt;
            &lt;style&gt;
                #map_e2a05881f4b10132fc1618d332747338 {
                    position: relative;
                    width: 100.0%;
                    height: 100.0%;
                    left: 0.0%;
                    top: 0.0%;
                }
                .leaflet-container { font-size: 1rem; }
            &lt;/style&gt;

            &lt;style&gt;html, body {
                width: 100%;
                height: 100%;
                margin: 0;
                padding: 0;
            }
            &lt;/style&gt;

            &lt;style&gt;#map {
                position:absolute;
                top:0;
                bottom:0;
                right:0;
                left:0;
                }
            &lt;/style&gt;

            &lt;script&gt;
                L_NO_TOUCH = false;
                L_DISABLE_3D = false;
            &lt;/script&gt;


&lt;/head&gt;
&lt;body&gt;


        &lt;div style=&quot;position:fixed;bottom:30px;left:30px;z-index:9999;
                    background:white;padding:14px 18px;border-radius:8px;
                    box-shadow:0 2px 8px rgba(0,0,0,.2);font-size:13px;
                    border-left:5px solid #f1c40f&quot;&gt;
          &lt;b&gt;Disaster Response Agent&lt;/b&gt;&lt;br&gt;&lt;br&gt;
          Threat level: &lt;b style=&quot;color:#f1c40f&quot;&gt;MEDIUM&lt;/b&gt;&lt;br&gt;
          Tsunami risk: &lt;b&gt;⚠ YES&lt;/b&gt;&lt;br&gt;
          Target city: &lt;b&gt;Honiara&lt;/b&gt;&lt;br&gt;
          Active hub: &lt;b&gt;MKQ&lt;/b&gt;&lt;br&gt;
          Route hops: &lt;b&gt;4&lt;/b&gt;&lt;br&gt;
          Route cost: &lt;b&gt;2,521 km-equiv&lt;/b&gt;&lt;br&gt;&lt;br&gt;
          &lt;span style=&quot;color:#f1c40f&quot;&gt;●&lt;/span&gt; Epicentre &amp;nbsp;
          ✈ Hub &amp;nbsp; 🏠 Target &amp;nbsp;
          &lt;span style=&quot;color:#f1c40f&quot;&gt;—&lt;/span&gt; Route
        &lt;/div&gt;


            &lt;div class=&quot;folium-map&quot; id=&quot;map_e2a05881f4b10132fc1618d332747338&quot; &gt;&lt;/div&gt;

&lt;/body&gt;
&lt;script&gt;


            var map_e2a05881f4b10132fc1618d332747338 = L.map(
                &quot;map_e2a05881f4b10132fc1618d332747338&quot;,
                {
                    center: [-9.614799999999999, 159.773],
                    crs: L.CRS.EPSG3857,
                    ...{
  &quot;zoom&quot;: 4,
  &quot;zoomControl&quot;: true,
  &quot;preferCanvas&quot;: false,
}

                }
            );





            var tile_layer_8f8f330593694848cd4e7db4624f4f01 = L.tileLayer(
                &quot;https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png&quot;,
                {
  &quot;minZoom&quot;: 0,
  &quot;maxZoom&quot;: 20,
  &quot;maxNativeZoom&quot;: 20,
  &quot;noWrap&quot;: false,
  &quot;attribution&quot;: &quot;\u0026copy; \u003ca href=\&quot;https://www.openstreetmap.org/copyright\&quot;\u003eOpenStreetMap\u003c/a\u003e contributors \u0026copy; \u003ca href=\&quot;https://carto.com/attributions\&quot;\u003eCARTO\u003c/a\u003e&quot;,
  &quot;subdomains&quot;: &quot;abcd&quot;,
  &quot;detectRetina&quot;: false,
  &quot;tms&quot;: false,
  &quot;opacity&quot;: 1,
}

            );


            tile_layer_8f8f330593694848cd4e7db4624f4f01.addTo(map_e2a05881f4b10132fc1618d332747338);


            var circle_marker_c939432532699aa83d65a2fb7dcec753 = L.circleMarker(
                [-9.7963, 159.596],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;#f1c40f&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;#f1c40f&quot;, &quot;fillOpacity&quot;: 0.35, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 18, &quot;stroke&quot;: true, &quot;weight&quot;: 2.5}
            ).addTo(map_e2a05881f4b10132fc1618d332747338);


        var popup_23d82a1aee1b0b9e563536007d6a41bc = L.popup({
  &quot;maxWidth&quot;: 250,
});



                var html_bd6a07f3b61226a381d1aab7b2a15dd5 = $(`&lt;div id=&quot;html_bd6a07f3b61226a381d1aab7b2a15dd5&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;&lt;b&gt;Earthquake epicentre&lt;/b&gt;&lt;br&gt;Magnitude: 7.0&lt;br&gt;Depth: 14.0 km&lt;br&gt;Threat: &lt;b&gt;MEDIUM&lt;/b&gt;&lt;br&gt;Tsunami: ⚠ YES&lt;/div&gt;`)[0];
                popup_23d82a1aee1b0b9e563536007d6a41bc.setContent(html_bd6a07f3b61226a381d1aab7b2a15dd5);



        circle_marker_c939432532699aa83d65a2fb7dcec753.bindPopup(popup_23d82a1aee1b0b9e563536007d6a41bc)
        ;




            circle_marker_c939432532699aa83d65a2fb7dcec753.bindTooltip(
                `&lt;div&gt;
                     Epicentre — MEDIUM threat
                 &lt;/div&gt;`,
                {
  &quot;sticky&quot;: true,
}
            );


            var marker_c1cce20a9b380e0e2908f7fa4b72ea18 = L.marker(
                [-9.7963, 159.596],
                {
}
            ).addTo(map_e2a05881f4b10132fc1618d332747338);


            var div_icon_302f72fe15c7f7c23892629ff895e8e4 = L.divIcon({
  &quot;html&quot;: &quot;\u003cdiv style=\&quot;font-size:20px;\&quot;\u003e\ud83d\udd34\u003c/div\u003e&quot;,
  &quot;iconSize&quot;: [30, 30],
  &quot;iconAnchor&quot;: [15, 15],
  &quot;className&quot;: &quot;empty&quot;,
});


                marker_c1cce20a9b380e0e2908f7fa4b72ea18.setIcon(div_icon_302f72fe15c7f7c23892629ff895e8e4);


            var marker_8941e46555665d8df08a4ce34b2aa796 = L.marker(
                [36.524824, 114.424126],
                {
}
            ).addTo(map_e2a05881f4b10132fc1618d332747338);


            var icon_c0b801ea45e69b34c7c5102b37282929 = L.AwesomeMarkers.icon(
                {
  &quot;markerColor&quot;: &quot;gray&quot;,
  &quot;iconColor&quot;: &quot;white&quot;,
  &quot;icon&quot;: &quot;plane&quot;,
  &quot;prefix&quot;: &quot;fa&quot;,
  &quot;extraClasses&quot;: &quot;fa-rotate-0&quot;,
}
            );


        var popup_5496b3455ddb5dac6c8a8c5afd9840ab = L.popup({
  &quot;maxWidth&quot;: 220,
});



                var html_f9c0b2cb25ed6144a531dde297fa65d0 = $(`&lt;div id=&quot;html_f9c0b2cb25ed6144a531dde297fa65d0&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;&lt;b&gt;Supply Hub 0&lt;/b&gt;&lt;br&gt;Handan Airport&lt;br&gt;IATA: HDG&lt;/div&gt;`)[0];
                popup_5496b3455ddb5dac6c8a8c5afd9840ab.setContent(html_f9c0b2cb25ed6144a531dde297fa65d0);



        marker_8941e46555665d8df08a4ce34b2aa796.bindPopup(popup_5496b3455ddb5dac6c8a8c5afd9840ab)
        ;




            marker_8941e46555665d8df08a4ce34b2aa796.bindTooltip(
                `&lt;div&gt;
                     Hub: HDG
                 &lt;/div&gt;`,
                {
  &quot;sticky&quot;: true,
}
            );


                marker_8941e46555665d8df08a4ce34b2aa796.setIcon(icon_c0b801ea45e69b34c7c5102b37282929);


            var marker_bb60eea8884d58f9211b3c4deb822079 = L.marker(
                [-8.523898, 140.419693],
                {
}
            ).addTo(map_e2a05881f4b10132fc1618d332747338);


            var icon_1e2d53f9ce68cba9b9443989bb3c16cf = L.AwesomeMarkers.icon(
                {
  &quot;markerColor&quot;: &quot;red&quot;,
  &quot;iconColor&quot;: &quot;white&quot;,
  &quot;icon&quot;: &quot;plane&quot;,
  &quot;prefix&quot;: &quot;fa&quot;,
  &quot;extraClasses&quot;: &quot;fa-rotate-0&quot;,
}
            );


        var popup_cc8a3bd20f898356ea17f888f3da10bf = L.popup({
  &quot;maxWidth&quot;: 220,
});



                var html_c6e3aa33b592d4fd6d1de8a201561b71 = $(`&lt;div id=&quot;html_c6e3aa33b592d4fd6d1de8a201561b71&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;&lt;b&gt;★ ACTIVE Supply Hub 1&lt;/b&gt;&lt;br&gt;Mopah International Airport&lt;br&gt;IATA: MKQ&lt;/div&gt;`)[0];
                popup_cc8a3bd20f898356ea17f888f3da10bf.setContent(html_c6e3aa33b592d4fd6d1de8a201561b71);



        marker_bb60eea8884d58f9211b3c4deb822079.bindPopup(popup_cc8a3bd20f898356ea17f888f3da10bf)
        ;




            marker_bb60eea8884d58f9211b3c4deb822079.bindTooltip(
                `&lt;div&gt;
                     ★ Active hub: MKQ
                 &lt;/div&gt;`,
                {
  &quot;sticky&quot;: true,
}
            );


                marker_bb60eea8884d58f9211b3c4deb822079.setIcon(icon_1e2d53f9ce68cba9b9443989bb3c16cf);


            var marker_d6e141240da8f8763069d8aa4a3cf185 = L.marker(
                [-27.165411, -109.421027],
                {
}
            ).addTo(map_e2a05881f4b10132fc1618d332747338);


            var icon_e25c4ed1197c2b1669d64c91238af885 = L.AwesomeMarkers.icon(
                {
  &quot;markerColor&quot;: &quot;gray&quot;,
  &quot;iconColor&quot;: &quot;white&quot;,
  &quot;icon&quot;: &quot;plane&quot;,
  &quot;prefix&quot;: &quot;fa&quot;,
  &quot;extraClasses&quot;: &quot;fa-rotate-0&quot;,
}
            );


        var popup_21e9fec9cf9ff1599acb5297e7e65848 = L.popup({
  &quot;maxWidth&quot;: 220,
});



                var html_75f9a95ad41eb87793a5b359072a4ab8 = $(`&lt;div id=&quot;html_75f9a95ad41eb87793a5b359072a4ab8&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;&lt;b&gt;Supply Hub 2&lt;/b&gt;&lt;br&gt;Mataveri International Airport&lt;br&gt;IATA: IPC&lt;/div&gt;`)[0];
                popup_21e9fec9cf9ff1599acb5297e7e65848.setContent(html_75f9a95ad41eb87793a5b359072a4ab8);



        marker_d6e141240da8f8763069d8aa4a3cf185.bindPopup(popup_21e9fec9cf9ff1599acb5297e7e65848)
        ;




            marker_d6e141240da8f8763069d8aa4a3cf185.bindTooltip(
                `&lt;div&gt;
                     Hub: IPC
                 &lt;/div&gt;`,
                {
  &quot;sticky&quot;: true,
}
            );


                marker_d6e141240da8f8763069d8aa4a3cf185.setIcon(icon_e25c4ed1197c2b1669d64c91238af885);


            var marker_45392462694417e663a13e93813abef4 = L.marker(
                [53.25429916379999, -131.813995361],
                {
}
            ).addTo(map_e2a05881f4b10132fc1618d332747338);


            var icon_92e87ae703a205044a06e35a871ec6b3 = L.AwesomeMarkers.icon(
                {
  &quot;markerColor&quot;: &quot;gray&quot;,
  &quot;iconColor&quot;: &quot;white&quot;,
  &quot;icon&quot;: &quot;plane&quot;,
  &quot;prefix&quot;: &quot;fa&quot;,
  &quot;extraClasses&quot;: &quot;fa-rotate-0&quot;,
}
            );


        var popup_e2fa06e1697fa0482aee171a0d3a8b38 = L.popup({
  &quot;maxWidth&quot;: 220,
});



                var html_7e7ab942ae2aa12c74f5de5ec4cfb1bf = $(`&lt;div id=&quot;html_7e7ab942ae2aa12c74f5de5ec4cfb1bf&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;&lt;b&gt;Supply Hub 3&lt;/b&gt;&lt;br&gt;Sandspit Airport&lt;br&gt;IATA: YZP&lt;/div&gt;`)[0];
                popup_e2fa06e1697fa0482aee171a0d3a8b38.setContent(html_7e7ab942ae2aa12c74f5de5ec4cfb1bf);



        marker_45392462694417e663a13e93813abef4.bindPopup(popup_e2fa06e1697fa0482aee171a0d3a8b38)
        ;




            marker_45392462694417e663a13e93813abef4.bindTooltip(
                `&lt;div&gt;
                     Hub: YZP
                 &lt;/div&gt;`,
                {
  &quot;sticky&quot;: true,
}
            );


                marker_45392462694417e663a13e93813abef4.setIcon(icon_92e87ae703a205044a06e35a871ec6b3);


            var marker_137879cd9ed1f29c13710e1e92b82798 = L.marker(
                [12.1415, -86.168198],
                {
}
            ).addTo(map_e2a05881f4b10132fc1618d332747338);


            var icon_edd4fc29ed02731c33a7e910f83c5da0 = L.AwesomeMarkers.icon(
                {
  &quot;markerColor&quot;: &quot;gray&quot;,
  &quot;iconColor&quot;: &quot;white&quot;,
  &quot;icon&quot;: &quot;plane&quot;,
  &quot;prefix&quot;: &quot;fa&quot;,
  &quot;extraClasses&quot;: &quot;fa-rotate-0&quot;,
}
            );


        var popup_f2b0713959ef47f7723aa9a6cf379033 = L.popup({
  &quot;maxWidth&quot;: 220,
});



                var html_aadd5d783c2b37c6738cdebd4c5bcc3b = $(`&lt;div id=&quot;html_aadd5d783c2b37c6738cdebd4c5bcc3b&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;&lt;b&gt;Supply Hub 4&lt;/b&gt;&lt;br&gt;Augusto C. Sandino (Managua) International Airport&lt;br&gt;IATA: MGA&lt;/div&gt;`)[0];
                popup_f2b0713959ef47f7723aa9a6cf379033.setContent(html_aadd5d783c2b37c6738cdebd4c5bcc3b);



        marker_137879cd9ed1f29c13710e1e92b82798.bindPopup(popup_f2b0713959ef47f7723aa9a6cf379033)
        ;




            marker_137879cd9ed1f29c13710e1e92b82798.bindTooltip(
                `&lt;div&gt;
                     Hub: MGA
                 &lt;/div&gt;`,
                {
  &quot;sticky&quot;: true,
}
            );


                marker_137879cd9ed1f29c13710e1e92b82798.setIcon(icon_edd4fc29ed02731c33a7e910f83c5da0);


            var marker_1ccd5f00808dba019f78d038bc8afef0 = L.marker(
                [-9.4333, 159.95],
                {
}
            ).addTo(map_e2a05881f4b10132fc1618d332747338);


            var icon_17127035ebba9dbc6ee1df1d0a3afa8e = L.AwesomeMarkers.icon(
                {
  &quot;markerColor&quot;: &quot;blue&quot;,
  &quot;iconColor&quot;: &quot;white&quot;,
  &quot;icon&quot;: &quot;home&quot;,
  &quot;prefix&quot;: &quot;fa&quot;,
  &quot;extraClasses&quot;: &quot;fa-rotate-0&quot;,
}
            );


        var popup_7ed7b3173565f460b7783937eee99e58 = L.popup({
  &quot;maxWidth&quot;: 250,
});



                var html_d1b892cfba7c1e21ae8f2335eec8f709 = $(`&lt;div id=&quot;html_d1b892cfba7c1e21ae8f2335eec8f709&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;&lt;b&gt;Target city: Honiara&lt;/b&gt;&lt;br&gt;Country: Solomon Islands&lt;br&gt;Population: 84,520&lt;br&gt;Nearest airport: Honiara International Airport&lt;/div&gt;`)[0];
                popup_7ed7b3173565f460b7783937eee99e58.setContent(html_d1b892cfba7c1e21ae8f2335eec8f709);



        marker_1ccd5f00808dba019f78d038bc8afef0.bindPopup(popup_7ed7b3173565f460b7783937eee99e58)
        ;




            marker_1ccd5f00808dba019f78d038bc8afef0.bindTooltip(
                `&lt;div&gt;
                     Target: Honiara
                 &lt;/div&gt;`,
                {
  &quot;sticky&quot;: true,
}
            );


                marker_1ccd5f00808dba019f78d038bc8afef0.setIcon(icon_17127035ebba9dbc6ee1df1d0a3afa8e);


            var poly_line_6d1a4dd24e3b492a98feb1d16fb5f654 = L.polyline(
                [[-8.523898, 140.419693], [-8.05000019073, 142.932998657], [-10.585636, 142.29277], [-9.428, 160.054993]],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;#f1c40f&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: false, &quot;fillColor&quot;: &quot;#f1c40f&quot;, &quot;fillOpacity&quot;: 0.2, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;noClip&quot;: false, &quot;opacity&quot;: 0.85, &quot;smoothFactor&quot;: 1.0, &quot;stroke&quot;: true, &quot;weight&quot;: 4}
            ).addTo(map_e2a05881f4b10132fc1618d332747338);


            poly_line_6d1a4dd24e3b492a98feb1d16fb5f654.bindTooltip(
                `&lt;div&gt;
                     A* route — 2,521 km-equiv
                 &lt;/div&gt;`,
                {
  &quot;sticky&quot;: true,
}
            );


            var circle_marker_81306cf23deae8db7a6d2ee2603b9c30 = L.circleMarker(
                [-8.05000019073, 142.932998657],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;white&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;#f1c40f&quot;, &quot;fillOpacity&quot;: 0.9, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 5, &quot;stroke&quot;: true, &quot;weight&quot;: 1.5}
            ).addTo(map_e2a05881f4b10132fc1618d332747338);


            circle_marker_81306cf23deae8db7a6d2ee2603b9c30.bindTooltip(
                `&lt;div&gt;
                     Via: OPU — Balimo Airport
                 &lt;/div&gt;`,
                {
  &quot;sticky&quot;: true,
}
            );


            var circle_marker_1c38d71dd7ee3cf17dad69982a69d99a = L.circleMarker(
                [-10.585636, 142.29277],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;white&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;#f1c40f&quot;, &quot;fillOpacity&quot;: 0.9, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 5, &quot;stroke&quot;: true, &quot;weight&quot;: 1.5}
            ).addTo(map_e2a05881f4b10132fc1618d332747338);


            circle_marker_1c38d71dd7ee3cf17dad69982a69d99a.bindTooltip(
                `&lt;div&gt;
                     Via: HID — Horn Island Airport
                 &lt;/div&gt;`,
                {
  &quot;sticky&quot;: true,
}
            );


            tile_layer_8f8f330593694848cd4e7db4624f4f01.addTo(map_e2a05881f4b10132fc1618d332747338);


                marker_c1cce20a9b380e0e2908f7fa4b72ea18.setIcon(div_icon_302f72fe15c7f7c23892629ff895e8e4);


                marker_8941e46555665d8df08a4ce34b2aa796.setIcon(icon_c0b801ea45e69b34c7c5102b37282929);


                marker_bb60eea8884d58f9211b3c4deb822079.setIcon(icon_1e2d53f9ce68cba9b9443989bb3c16cf);


                marker_d6e141240da8f8763069d8aa4a3cf185.setIcon(icon_e25c4ed1197c2b1669d64c91238af885);


                marker_45392462694417e663a13e93813abef4.setIcon(icon_92e87ae703a205044a06e35a871ec6b3);


                marker_137879cd9ed1f29c13710e1e92b82798.setIcon(icon_edd4fc29ed02731c33a7e910f83c5da0);


                marker_1ccd5f00808dba019f78d038bc8afef0.setIcon(icon_17127035ebba9dbc6ee1df1d0a3afa8e);

&lt;/script&gt;
&lt;/html&gt;" style="position:absolute;width:100%;height:100%;left:0;top:0;border:none !important;" allowfullscreen webkitallowfullscreen mozallowfullscreen></iframe></div></div>



## Run on Multiple Events — Stress Test


```python
# Pick 4 real earthquakes from the dataset spanning different regions/threats
test_events = [
    # M7.3 Tonga
    {'latitude': -19.2918, 'longitude': -172.129, 'magnitude': 7.3, 'depth': 37.0,
     'cdi': 5, 'mmi': 5, 'sig': 833, 'nst': 149, 'dmin': 1.865, 'gap': 21.0,
     'month_sin': np.sin(2*np.pi*11/12), 'month_cos': np.cos(2*np.pi*11/12),
     'year': 2022, 'magType_mww': 1},
    # M6.6 deep event
    {'latitude': -25.5948, 'longitude': 178.278, 'magnitude': 6.6, 'depth': 624.464,
     'cdi': 0, 'mmi': 2, 'sig': 670, 'nst': 131, 'dmin': 4.998, 'gap': 27.0,
     'month_sin': np.sin(2*np.pi*9/12), 'month_cos': np.cos(2*np.pi*9/12),
     'year': 2022, 'magType_mww': 1},
    # M9.1 historic-scale (Sumatra-like)
    {'latitude': 3.295,    'longitude': 95.982,   'magnitude': 9.1, 'depth': 30.0,
     'cdi': 9, 'mmi': 9, 'sig': 2910, 'nst': 500, 'dmin': 0.1,  'gap': 10.0,
     'month_sin': np.sin(2*np.pi*12/12),'month_cos': np.cos(2*np.pi*12/12),
     'year': 2004, 'magType_mww': 1},
]
 
results_summary = []
for i, evt in enumerate(test_events):
    print(f"\n{'─'*55}")
    print(f"  TEST EVENT {i+1}: lat={evt['latitude']}, lon={evt['longitude']}, mag={evt['magnitude']}")
    perception   = agent.perceive(evt)
    goal_city    = agent.formulate_goal(evt)
    hub_row, goal_apt, goal_city, nodes, path, cost = agent.plan_route(evt, goal_city)
    agent.act(perception, hub_row, goal_city, goal_apt, nodes, path, cost)
 
    route_cost = round(cost, 0) if path else "Unreachable"
 
    results_summary.append({
        'event'        : f"M{evt['magnitude']} lat={evt['latitude']}",
        'threat'       : perception['threat_level'],
        'tsunami'      : perception['tsunami_prediction'],
        'target_city'  : goal_city['city'],
        'hub'          : hub_row['iata_code'],
        'route_hops'   : len(path) if path else 0,
        'route_cost_km': route_cost,
    })
 
print("\n\nSUMMARY TABLE")
print(pd.DataFrame(results_summary).to_string(index=False))
```

    
    ───────────────────────────────────────────────────────
      TEST EVENT 1: lat=-19.2918, lon=-172.129, mag=7.3
    
    ============================================================
      DISASTER RESPONSE AGENT — DISPATCH ORDER
    ============================================================
      Threat level   : HIGH
      Tsunami risk   : YES  (prob=100.0%)
      Target city    : Nuku‘alofa (Tonga)
      Nearest airport: Fua'amotu International Airport
      Supply hub     : Mopah International Airport [MKQ]
      Dispatch mode  : AIR DISPATCH (cargo / charter flights)
      Route hops     : 7 airports
      Route cost     : 9,171 km-equiv
    
      Supplies dispatched: 8 cargo flights + 3 medical teams + field hospital (150t)
      + Coastal rescue units + tsunami relief kits
    ============================================================
    
      ROUTE:
        START → [MKQ] Mopah International Airport
          →     [KMA] Kerema Airport
          →     [PNP] Girua Airport
          →     [SON] Santo Pekoa International Airport
          →     [NAN] Nadi International Airport
          →     [SUV] Nausori International Airport
        → END   [TBU] Fua'amotu International Airport
    
    ───────────────────────────────────────────────────────
      TEST EVENT 2: lat=-25.5948, lon=178.278, mag=6.6
    
    ============================================================
      DISASTER RESPONSE AGENT — DISPATCH ORDER
    ============================================================
      Threat level   : LOW
      Tsunami risk   : YES  (prob=84.1%)
      Target city    : Nuku‘alofa (Tonga)
      Nearest airport: Fua'amotu International Airport
      Supply hub     : Mopah International Airport [MKQ]
      Dispatch mode  : AIR DISPATCH (cargo / charter flights)
      Route hops     : 7 airports
      Route cost     : 9,171 km-equiv
    
      Supplies dispatched: 2 cargo flights (basic supplies, 20t)
      + Coastal rescue units + tsunami relief kits
    ============================================================
    
      ROUTE:
        START → [MKQ] Mopah International Airport
          →     [KMA] Kerema Airport
          →     [PNP] Girua Airport
          →     [SON] Santo Pekoa International Airport
          →     [NAN] Nadi International Airport
          →     [SUV] Nausori International Airport
        → END   [TBU] Fua'amotu International Airport
    
    ───────────────────────────────────────────────────────
      TEST EVENT 3: lat=3.295, lon=95.982, mag=9.1
      [PHYSICS OVERRIDE] M9.1 shallow (30.0km) over water — overriding to YES.
    
    ============================================================
      DISASTER RESPONSE AGENT — DISPATCH ORDER
    ============================================================
      Threat level   : CRITICAL
      Tsunami risk   : YES  (prob=100.0%)
      Target city    : Meulaboh (Indonesia)
      Nearest airport: Cut Nyak Dhien Airport
      Supply hub     : Handan Airport [HDG]
      Dispatch mode  : AIR DISPATCH (cargo / charter flights)
      Route hops     : 3 airports
      Route cost     : 4,063 km-equiv
    
      Supplies dispatched: 20 cargo flights + 10 medical teams + 2 field hospitals (500t)
      + Coastal rescue units + tsunami relief kits
    ============================================================
    
      ROUTE:
        START → [HDG] Handan Airport
          →     [VTE] Wattay International Airport
        → END   [MEQ] Cut Nyak Dhien Airport
    
    
    SUMMARY TABLE
                event   threat  tsunami target_city hub  route_hops  route_cost_km
    M7.3 lat=-19.2918     HIGH        1  Nuku‘alofa MKQ           7         9171.0
    M6.6 lat=-25.5948      LOW        1  Nuku‘alofa MKQ           7         9171.0
       M9.1 lat=3.295 CRITICAL        1    Meulaboh HDG           3         4063.0


## Agent Architecture Diagram


```python
print("""
╔══════════════════════════════════════════════════════════╗
║         GOAL-BASED AGENT ARCHITECTURE                    ║
╠══════════════════════════════════════════════════════════╣
║  ENVIRONMENT   Earthquake sensor (lat, lon, mag, depth)  ║
╠══════════════════════════════════════════════════════════╣
║  PERCEPT       Raw earthquake reading                    ║
║  PERCEIVE      Tsunami classifier (binary, pkl)          ║
║                Threat classifier  (4-class, pkl)         ║
╠══════════════════════════════════════════════════════════╣
║  GOAL          Nearest city ≥ 10k pop to epicentre       ║
╠══════════════════════════════════════════════════════════╣
║  PLAN          1. Select nearest supply hub              ║
║                2. Find goal city airport                 ║
║                3. Build local airport graph              ║
║                4. A* (cost = dist × land/water mult.)    ║
╠══════════════════════════════════════════════════════════╣
║  ACT           Print dispatch order                      ║
║                Folium interactive map                    ║
╚══════════════════════════════════════════════════════════╝
""")
```

    
    ╔══════════════════════════════════════════════════════════╗
    ║         GOAL-BASED AGENT ARCHITECTURE                    ║
    ╠══════════════════════════════════════════════════════════╣
    ║  ENVIRONMENT   Earthquake sensor (lat, lon, mag, depth)  ║
    ╠══════════════════════════════════════════════════════════╣
    ║  PERCEPT       Raw earthquake reading                    ║
    ║  PERCEIVE      Tsunami classifier (binary, pkl)          ║
    ║                Threat classifier  (4-class, pkl)         ║
    ╠══════════════════════════════════════════════════════════╣
    ║  GOAL          Nearest city ≥ 10k pop to epicentre       ║
    ╠══════════════════════════════════════════════════════════╣
    ║  PLAN          1. Select nearest supply hub              ║
    ║                2. Find goal city airport                 ║
    ║                3. Build local airport graph              ║
    ║                4. A* (cost = dist × land/water mult.)    ║
    ╠══════════════════════════════════════════════════════════╣
    ║  ACT           Print dispatch order                      ║
    ║                Folium interactive map                    ║
    ╚══════════════════════════════════════════════════════════╝
    

