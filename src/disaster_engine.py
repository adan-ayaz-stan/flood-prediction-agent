import numpy as np
import pandas as pd
import folium
import joblib
import heapq
import warnings
from global_land_mask import globe
from folium.plugins import HeatMap

import os
import csv
from datetime import datetime

warnings.filterwarnings('ignore')

# ==========================================
# 1. PATH CONFIGURATION & MODEL LOADING
# ==========================================
MODEL_DIR = './models/'
DATA_DIR  = './data/'

def load_all_assets():
    """Loads all models, scalers, and datasets. Called once by Streamlit cache."""
    assets = {}
    
    # ML Models & Scalers
    assets['tsunami_model']    = joblib.load(MODEL_DIR + 'tsunami_model.pkl')
    assets['tsunami_scaler']   = joblib.load(MODEL_DIR + 'tsunami_scaler.pkl')
    assets['tsunami_features'] = joblib.load(MODEL_DIR + 'tsunami_features.pkl')
    
    assets['threat_model']    = joblib.load(MODEL_DIR + 'threat_model.pkl')
    assets['threat_scaler']   = joblib.load(MODEL_DIR + 'threat_scaler.pkl')
    assets['threat_le']       = joblib.load(MODEL_DIR + 'threat_label_encoder.pkl')
    assets['threat_features'] = joblib.load(MODEL_DIR + 'threat_features.pkl')
    
    # Datasets
    assets['hubs_df'] = pd.read_csv(MODEL_DIR + 'supply_hubs.csv')
    
    apt = pd.read_csv(DATA_DIR + 'airports.csv')
    assets['apt_clean'] = (
        apt[apt['type'].isin(['large_airport', 'medium_airport', 'small_airport'])]
        .dropna(subset=['latitude_deg', 'longitude_deg'])
        .rename(columns={'latitude_deg': 'lat', 'longitude_deg': 'lon'})
        [['name', 'iata_code', 'type', 'lat', 'lon', 'iso_country', 'municipality']]
        .reset_index(drop=True)
    )
    
    cit = pd.read_csv(DATA_DIR + 'worldcities.csv')
    cit_clean = cit.dropna(subset=['lat', 'lng', 'population']).copy()
    cit_clean['lat']        = pd.to_numeric(cit_clean['lat'], errors='coerce')
    cit_clean['lng']        = pd.to_numeric(cit_clean['lng'], errors='coerce')
    cit_clean['population'] = pd.to_numeric(cit_clean['population'], errors='coerce')
    assets['cit_clean'] = cit_clean.dropna(subset=['lat', 'lng']).reset_index(drop=True)
    
    assets['historical_quakes'] = pd.read_csv(DATA_DIR + 'earthquake_data.csv').dropna(subset=['latitude', 'longitude', 'magnitude'])
    
    return assets

# ==========================================
# 2. MATH & GEOGRAPHY HELPERS
# ==========================================
def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    rlat1, rlon1 = np.radians(np.float64(lat1)), np.radians(np.float64(lon1))
    rlat2, rlon2 = np.radians(np.asarray(lat2, dtype=np.float64)), np.radians(np.asarray(lon2, dtype=np.float64))
    dlat, dlon = rlat2 - rlat1, rlon2 - rlon1
    a = np.sin(dlat / 2)**2 + np.cos(rlat1) * np.cos(rlat2) * np.sin(dlon / 2)**2
    return np.squeeze(2 * R * np.arcsin(np.sqrt(np.clip(a, 0, 1))))

def is_land(lat, lon):
    return bool(globe.is_land(float(lat), float(lon)))

def edge_cost(lat1, lon1, lat2, lon2):
    dist = float(np.squeeze(haversine_km(lat1, lon1, [lat2], [lon2])))
    mid_land = is_land((lat1 + lat2) / 2, (lon1 + lon2) / 2)
    src_land, dst_land = is_land(lat1, lon1), is_land(lat2, lon2)
    
    if src_land and dst_land and mid_land: mult = 1.0
    elif not mid_land: mult = 2.5
    else: mult = 1.5
    return dist * mult

def nearest_airports(lat, lon, apt_df, k=40):
    dists = haversine_km(lat, lon, apt_df['lat'].values, apt_df['lon'].values)
    idx = np.argsort(dists)[:k]
    result = apt_df.iloc[idx].copy()
    result['dist_km'] = dists[idx]
    return result.reset_index(drop=True)

def nearest_city(lat, lon, city_df, min_pop=10_000):
    pop = city_df[city_df['population'] >= min_pop].copy()
    dists = haversine_km(lat, lon, pop['lat'].values, pop['lng'].values)
    pop['dist_km'] = dists
    return pop.nsmallest(1, 'dist_km').iloc[0]

def nearest_hub(goal_lat, goal_lon, hubs):
    dists = haversine_km(goal_lat, goal_lon, hubs['hub_lat'].values, hubs['hub_lon'].values)
    return hubs.iloc[np.argmin(dists)].copy()

# ==========================================
# 3. GRAPH & A* ALGORITHM
# ==========================================
MAX_EDGE_KM = 4000

def build_graph(hub_row, goal_apt_row, apt_df):
    hub_lat, hub_lon = hub_row['hub_lat'], hub_row['hub_lon']
    goal_lat, goal_lon = goal_apt_row['lat'], goal_apt_row['lon']

    near_hub  = nearest_airports(hub_lat, hub_lon, apt_df, k=40)
    near_goal = nearest_airports(goal_lat, goal_lon, apt_df, k=30)
    
    mid_lat, mid_lon = (hub_lat + goal_lat) / 2, (hub_lon + goal_lon) / 2
    stepping_stones  = nearest_airports(mid_lat, mid_lon, apt_df, k=20)

    candidates = pd.concat([near_hub, stepping_stones, near_goal], ignore_index=True)

    hub_node  = {'name': hub_row['airport_name'], 'iata': str(hub_row.get('iata_code', '---')), 'lat': hub_lat, 'lon': hub_lon}
    goal_node = {'name': goal_apt_row['name'], 'iata': str(goal_apt_row.get('iata_code', '---')), 'lat': goal_lat, 'lon': goal_lon}

    nodes = [hub_node]
    seen  = {(round(hub_lat, 2), round(hub_lon, 2))}

    for _, r in candidates.iterrows():
        key = (round(r['lat'], 2), round(r['lon'], 2))
        if key not in seen:
            seen.add(key)
            nodes.append({'name': r['name'], 'iata': str(r['iata_code']), 'lat': r['lat'], 'lon': r['lon']})

    goal_key = (round(goal_lat, 2), round(goal_lon, 2))
    if goal_key not in seen:
        nodes.append(goal_node)

    start_idx = 0
    goal_idx  = next((i for i, n in enumerate(nodes) if n['name'] == goal_node['name']), len(nodes) - 1)

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
    def h(idx):
        return float(np.squeeze(haversine_km(nodes[idx]['lat'], nodes[idx]['lon'], [nodes[goal]['lat']], [nodes[goal]['lon']])))
    
    open_set = [(h(start), 0.0, start, [start])]
    visited  = {}
    
    while open_set:
        f, g, current, path = heapq.heappop(open_set)
        if current == goal: return path, g
        if current in visited and visited[current] <= g: continue
        visited[current] = g
        
        for neighbour, cost in adj[current]:
            new_g = g + cost
            heapq.heappush(open_set, (new_g + h(neighbour), new_g, neighbour, path + [neighbour]))
            
    return None, float('inf')

# ==========================================
# 4. NEW: KNOWLEDGE BASE ENGINE (FORWARD CHAINING)
# ==========================================
class ForwardChainingEngine:
    def __init__(self, rules):
        self.rules = rules

    def infer(self, initial_facts):
        inferred_facts = set(initial_facts)
        trace_log = []
        
        changed = True
        while changed:
            changed = False
            for rule in self.rules:
                if all(condition in inferred_facts for condition in rule['if']):
                    if rule['then'] not in inferred_facts:
                        inferred_facts.add(rule['then'])
                        trace_log.append(f"Fired Rule: {rule['name']} -> '{rule['then']}'")
                        changed = True
                        
        return inferred_facts, trace_log

# 13 Expert Domain Rules for Earthquake Response
EARTHQUAKE_RULES = [
    {"name": "Rule 1", "if": ["high_magnitude", "shallow_depth"], "then": "severe_surface_shaking"},
    {"name": "Rule 2", "if": ["low_magnitude", "low_population_density"], "then": "predicted_threat_level_is_low"},
    {"name": "Rule 3", "if": ["over_water", "severe_surface_shaking"], "then": "tsunami_warning_active"},
    {"name": "Rule 4", "if": ["high_magnitude", "inland"], "then": "severe_infrastructure_damage"},
    {"name": "Rule 5", "if": ["severe_surface_shaking", "high_population_density"], "then": "mass_casualties_expected"},
    {"name": "Rule 6", "if": ["severe_infrastructure_damage", "high_population_density"], "then": "widespread_homelessness"},
    {"name": "Rule 7", "if": ["predicted_threat_level_is_low"], "then": "deploy_basic_supplies"},
    {"name": "Rule 8", "if": ["mass_casualties_expected"], "then": "deploy_field_hospital"},
    {"name": "Rule 9", "if": ["tsunami_warning_active"], "then": "deploy_navy_rescue_crafts"},
    {"name": "Rule 10", "if": ["tsunami_warning_active"], "then": "initiate_coastal_evacuation_protocols"},
    {"name": "Rule 11", "if": ["severe_infrastructure_damage"], "then": "deploy_heavy_rubble_clearance_vehicles"},
    {"name": "Rule 12", "if": ["widespread_homelessness"], "then": "establish_displaced_persons_camp"},
    {"name": "Rule 13", "if": ["mass_casualties_expected", "severe_infrastructure_damage"], "then": "request_international_medical_aid"}
]


# ==========================================
# 5. DISASTER AGENT CLASS
# ==========================================
class DisasterResponseAgent:
    
    THREAT_COLORS = {
        'LOW': '#2ecc71', 'MEDIUM': '#f1c40f', 'HIGH': '#e67e22', 'CRITICAL': '#e74c3c'
    }
    
    THREAT_SUPPLIES_LAND = {
        'LOW': '2 trucks (20t)', 'MEDIUM': '5 trucks + 1 med team (50t)',
        'HIGH': '10 trucks + 3 med teams + 1 hospital (150t)', 'CRITICAL': '30 trucks + 10 med teams + 2 hospitals (500t)'
    }
    THREAT_SUPPLIES_AIR = {
        'LOW': '2 cargo flights (20t)', 'MEDIUM': '4 cargo flights + 1 med team (50t)',
        'HIGH': '8 cargo flights + 3 med teams + 1 hospital (150t)', 'CRITICAL': '20 cargo flights + 10 med teams + 2 hospitals (500t)'
    }
    THREAT_SUPPLIES_MARITIME = {
        'LOW': '1 naval vessel', 'MEDIUM': '2 naval vessels + med team',
        'HIGH': '4 naval vessels + hospital ship + rescue', 'CRITICAL': 'Naval task force + hospital ship + 10 rescue teams'
    }

    def __init__(self, assets):
        self.assets = assets
        self.hubs = assets['hubs_df']
        self.airports = assets['apt_clean']
        self.cities = assets['cit_clean']
        self.historical_quakes = assets['historical_quakes']
        
        # === NEW: Initialize Knowledge Base ===
        self.kbs_engine = ForwardChainingEngine(EARTHQUAKE_RULES)

    def _build_model_row(self, event, features):
        row = {f: event.get(f, 0) for f in features}
        df_row = pd.DataFrame([row])
        for col in features:
            if col not in df_row.columns: df_row[col] = 0
        return df_row[features]

    def _classify_dispatch_mode(self, path):
        if path is None or len(path) == 0: return 'maritime'
        return 'air'

    def perceive(self, event):
        # Tsunami ML
        t_row = self._build_model_row(event, self.assets['tsunami_features'])
        t_scaled = self.assets['tsunami_scaler'].transform(t_row)
        ml_pred = self.assets['tsunami_model'].predict(t_scaled)[0]
        t_prob = self.assets['tsunami_model'].predict_proba(t_scaled)[0][1] if hasattr(self.assets['tsunami_model'], 'predict_proba') else 0.0
        
        # Physics Override (Leaves original ML prob intact for the charts)
        mag, depth, lat, lon = event.get('magnitude', 0), event.get('depth', 999), event.get('latitude', 0), event.get('longitude', 0)
        over_water = not is_land(lat, lon)
        final_t_pred = 1 if ((mag >= 8.5 and depth <= 70 and over_water) or mag >= 9.0) else ml_pred
            
        # Threat ML
        th_row = self._build_model_row(event, self.assets['threat_features'])
        th_scaled = self.assets['threat_scaler'].transform(th_row)
        th_pred = self.assets['threat_model'].predict(th_scaled)[0]
        threat_label = self.assets['threat_le'].inverse_transform([th_pred])[0]
        
        # Extract Threat Probabilities for the UI Chart
        th_probs = {}
        if hasattr(self.assets['threat_model'], 'predict_proba'):
            raw_probs = self.assets['threat_model'].predict_proba(th_scaled)[0]
            classes = self.assets['threat_le'].classes_
            th_probs = {classes[i]: raw_probs[i] for i in range(len(classes))}

        return {
            'tsunami_prediction': final_t_pred,
            'tsunami_probability': round(t_prob, 3), # Raw ML probability
            'threat_level': threat_label,
            'threat_probabilities': th_probs         # Raw Threat probabilities
        }

    def formulate_goal(self, event):
        return nearest_city(event['latitude'], event['longitude'], self.cities)

    def plan_route(self, event, goal_city):
        hub_row = nearest_hub(goal_city['lat'], goal_city['lng'], self.hubs)
        goal_apt = nearest_airports(goal_city['lat'], goal_city['lng'], self.airports, k=1).iloc[0]
        nodes, adj, start_idx, goal_idx = build_graph(hub_row, goal_apt, self.airports)
        path, cost = astar(nodes, adj, start_idx, goal_idx)
        return hub_row, goal_apt, nodes, path, cost
    
    # === NEW: Telemetry Mapper for KBS ===
    def _map_telemetry_to_facts(self, event, perception, goal_city):
        """Translates raw model data into KBS logic facts."""
        facts = set()
        
        if event.get('magnitude', 0) >= 7.0: facts.add("high_magnitude")
        elif event.get('magnitude', 0) <= 5.5: facts.add("low_magnitude")
            
        if event.get('depth', 999) < 30.0: facts.add("shallow_depth")
            
        if not is_land(event.get('latitude', 0), event.get('longitude', 0)): facts.add("over_water")
        else: facts.add("inland")
            
        if goal_city.get('population', 0) >= 500000: facts.add("high_population_density")
        else: facts.add("low_population_density")
            
        if perception['threat_level'] == 'LOW': facts.add("predicted_threat_level_is_low")
        if perception['tsunami_prediction'] == 1: facts.add("tsunami_warning_active")
            
        return facts

    # === NEW: Execute KBS ===
    def run_knowledge_base(self, event, perception, goal_city):
        """Runs the Forward Chaining Engine and extracts operational protocols."""
        initial_facts = self._map_telemetry_to_facts(event, perception, goal_city)
        final_facts, trace = self.kbs_engine.infer(initial_facts)
        
        # Clean up the output string to easily display on frontend
        actionable_deployments = [
            fact.replace('_', ' ').title() 
            for fact in final_facts 
            if fact.startswith(("deploy_", "request_", "initiate_", "establish_"))
        ]
        
        return trace, actionable_deployments

    def get_dispatch_details(self, perception, path):
        threat = perception['threat_level']
        mode = self._classify_dispatch_mode(path)
        
        if mode == 'maritime':
            supplies = self.THREAT_SUPPLIES_MARITIME[threat]
            mode_str = 'MARITIME DISPATCH (No air bridge available)'
        elif mode == 'air':
            supplies = self.THREAT_SUPPLIES_AIR[threat]
            mode_str = 'AIR DISPATCH (Cargo / charter flights)'
        else:
            supplies = self.THREAT_SUPPLIES_LAND[threat]
            mode_str = 'LAND DISPATCH (Trucks)'
            
        if perception['tsunami_prediction']:
            supplies += " + Coastal Rescue Units + Tsunami Relief Kits"
            
        return mode, mode_str, supplies
    
    def log_dispatch(self, event, perception, goal_city, hub_row, mode_str, cost, file_path='./data/dispatch_history.csv'):
        import os
        import csv
        from datetime import datetime
        
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        file_exists = os.path.isfile(file_path)
        
        headers = [
            "timestamp", "latitude", "longitude", "magnitude", "depth_km", 
            "threat_level", "tsunami_risk", "target_city", "hub_iata", 
            "dispatch_mode", "route_cost_km",
            "prob_tsunami", "prob_threat_LOW", "prob_threat_MEDIUM", 
            "prob_threat_HIGH", "prob_threat_CRITICAL"
        ]
        
        th_probs = perception.get('threat_probabilities', {})
        
        row = [
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            round(event['latitude'], 4), round(event['longitude'], 4), 
            event['magnitude'], event['depth'],
            perception['threat_level'],
            'YES' if perception['tsunami_prediction'] else 'NO',
            f"{goal_city['city']} ({goal_city['country']})",
            hub_row['iata_code'], mode_str,
            round(cost, 0) if cost != float('inf') else 'Unreachable',
            perception.get('tsunami_probability', 0.0),
            round(th_probs.get('LOW', 0.0), 3), round(th_probs.get('MEDIUM', 0.0), 3),
            round(th_probs.get('HIGH', 0.0), 3), round(th_probs.get('CRITICAL', 0.0), 3)
        ]
        
        with open(file_path, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(headers)
            writer.writerow(row)

    def display_map(self, event, perception, hub_row, goal_city, goal_apt, nodes, path, cost, mode, show_heatmap=False):
        threat = perception['threat_level']
        threat_color = self.THREAT_COLORS[threat]
        map_center = [(event['latitude'] + goal_city['lat']) / 2, (event['longitude'] + goal_city['lng']) / 2]

        m = folium.Map(location=map_center, zoom_start=4, tiles='CartoDB positron')

        if show_heatmap:
            significant_quakes = self.historical_quakes[self.historical_quakes['magnitude'] >= 5.0]
            heat_data = significant_quakes[['latitude', 'longitude', 'magnitude']].values.tolist()
            HeatMap(heat_data, radius=12, blur=15, max_zoom=1).add_to(m)

        folium.CircleMarker(
            location=[event['latitude'], event['longitude']], radius=18, color=threat_color, fill=True, fill_opacity=0.3, weight=2.5,
            tooltip=f"Epicentre — {threat}"
        ).add_to(m)

        for _, hub in self.hubs.iterrows():
            is_active = hub['hub_id'] == hub_row['hub_id']
            color = 'blue' if is_active else 'lightgray'
            icon_type = 'star' if is_active else 'plane'
            
            folium.Marker(
                location=[hub['hub_lat'], hub['hub_lon']], 
                icon=folium.Icon(color=color, icon=icon_type, prefix='fa'),
                tooltip=f"{'🌟 ACTIVE HUB: ' if is_active else 'Standby Hub: '}{hub['iata_code']}"
            ).add_to(m)

        folium.Marker(
            location=[goal_city['lat'], goal_city['lng']], icon=folium.Icon(color='red', icon='home', prefix='fa'),
            tooltip=f"Target: {goal_city['city']}"
        ).add_to(m)

        if path and len(path) > 1:
            route_coords = [[nodes[i]['lat'], nodes[i]['lon']] for i in path]
            folium.PolyLine(locations=route_coords, color=threat_color, weight=4, opacity=0.85).add_to(m)
            for node_idx in path[1:-1]:
                n = nodes[node_idx]
                folium.CircleMarker(location=[n['lat'], n['lon']], radius=5, color='white', fill=True, fill_color=threat_color, fill_opacity=0.9, weight=1.5, tooltip=n['iata']).add_to(m)
        elif mode == 'maritime':
            folium.PolyLine(locations=[[hub_row['hub_lat'], hub_row['hub_lon']], [goal_city['lat'], goal_city['lng']]], color='navy', weight=3, opacity=0.5, dash_array='8 6').add_to(m)

        return m
    
    def generate_base_map(self):
        m = folium.Map(location=[20, 0], zoom_start=2, tiles='CartoDB positron')
        for _, hub in self.hubs.iterrows():
            folium.Marker(
                location=[hub['hub_lat'], hub['hub_lon']],
                icon=folium.Icon(color='lightgray', icon='plane', prefix='fa'),
                tooltip=f"Standby Hub: {hub['airport_name']} [{hub['iata_code']}]"
            ).add_to(m)
            
        return m