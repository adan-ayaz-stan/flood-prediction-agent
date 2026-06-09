import streamlit as st
import folium
from streamlit_folium import st_folium
import numpy as np
from datetime import datetime

import plotly.express as px
import plotly.graph_objects as go

# Import our backend agent
from disaster_engine import DisasterResponseAgent, load_all_assets

st.set_page_config(page_title="Disaster Response AI", layout="wide", initial_sidebar_state="expanded")

# 1. INITIALIZE CACHE & STATE
@st.cache_resource
def initialize_agent():
    assets = load_all_assets()
    return DisasterResponseAgent(assets)

with st.spinner("Loading AI Models and Global Maps..."):
    agent = initialize_agent()

# Create memory variables so the app doesn't forget its state during reruns
if 'processed_click' not in st.session_state:
    st.session_state.processed_click = None
if 'final_map' not in st.session_state:
    st.session_state.final_map = None
if 'dashboard_data' not in st.session_state:
    st.session_state.dashboard_data = None

# 2. UI LAYOUT
st.title("🌍 Disaster Response Agent Dashboard")
st.markdown("Click anywhere on the map to simulate an earthquake. The AI will instantly calculate the risk and deploy a supply route.")

col1, col2 = st.columns([2, 1])

with st.sidebar:
    st.header("Seismic Parameters")
    magnitude = st.slider("Magnitude", min_value=5.0, max_value=9.5, value=7.5, step=0.1)
    depth = st.slider("Depth (km)", min_value=0, max_value=600, value=30, step=10)
    
    st.markdown("---")
    st.header("Map Layers")
    show_heatmap = st.toggle("🔥 Show Seismic Heatmap", value=False)

# 3. INTERACTIVE MAP (Col 1)
with col1:
    st.subheader("1. Simulation Map")
    
    # NEW: Render the agent's base map with hubs if no event has happened yet
    if st.session_state.final_map:
        map_to_render = st.session_state.final_map
    else:
        map_to_render = agent.generate_base_map()
    
    map_data = st_folium(map_to_render, height=600, width=800, key="main_map", returned_objects=['last_clicked'])

# 4. EXECUTION PIPELINE (Col 2)
with col2:
    st.subheader("2. Live Dispatch Board")
    
    current_click = map_data.get('last_clicked')
    
    # --- THE FIX: Only run the heavy math if this is a BRAND NEW click ---
    if current_click is not None and current_click != st.session_state.processed_click:
        
        # Instantly update our memory so we don't get caught in an infinite loop
        st.session_state.processed_click = current_click
        
        lat, lon = current_click['lat'], current_click['lng']
        current_month = datetime.now().month
        
        dynamic_mmi = min(10, max(1, int(magnitude)))
        dynamic_sig = max(100, int((magnitude - 4) * 300))
        current_month = datetime.now().month
        
        simulated_event = {
            'latitude': lat, 'longitude': lon, 
            'magnitude': magnitude, 'depth': depth,
            'cdi': dynamic_mmi, 'mmi': dynamic_mmi, 'sig': dynamic_sig, 
            'nst': 150, 'dmin': 1.0, 'gap': 20.0,
            'month_sin': np.sin(2 * np.pi * current_month / 12), 
            'month_cos': np.cos(2 * np.pi * current_month / 12),
            'year': datetime.now().year, 'magType_mww': 1
        }
        
        with st.spinner('Agent is analyzing and planning route...'):
            perception = agent.perceive(simulated_event)
            goal_city  = agent.formulate_goal(simulated_event)
            hub_row, goal_apt, nodes, path, cost = agent.plan_route(simulated_event, goal_city)
            mode, mode_str, supplies = agent.get_dispatch_details(perception, path)
            
            trace_logs, actionable_deployments = agent.run_knowledge_base(simulated_event, perception, goal_city)
            
            agent.log_dispatch(simulated_event, perception, goal_city, hub_row, mode_str, cost)
            
            # Save the new map to state
            st.session_state.final_map = agent.display_map(
                simulated_event, perception, hub_row, goal_city, goal_apt, nodes, path, cost, mode, show_heatmap
            )
            
            # Save the text readouts to state
            st.session_state.dashboard_data = {
                'perception': perception,
                'threat': perception['threat_level'],
                'tsunami': perception['tsunami_prediction'],
                'target': f"{goal_city['city']} ({goal_city['country']})",
                'hub': f"{hub_row['airport_name']} [{hub_row['iata_code']}]",
                'path': path, 'cost': cost, 'mode_str': mode_str, 'supplies': supplies,
                'trace_logs': trace_logs,
                'actionable_deployments': actionable_deployments
            }
            
        # Rerun the app instantly to display the new map smoothly
        st.rerun()

    # --- DISPLAY THE RESULTS ---
    # Because we use session_state, the text won't vanish when the map renders!
    if st.session_state.dashboard_data:
        d = st.session_state.dashboard_data
        st.success("Target Locked. Route Calculated.")
        st.write(f"**Threat Level:** `{d['threat']}`")
        st.write(f"**Tsunami Risk:** `{'🚨 YES' if d['tsunami'] else '✅ NO'}`")
        
        st.markdown("---")
        st.write(f"**📍 Target City:** {d['target']}")
        st.write(f"**✈️ Nearest Hub:** {d['hub']}")
        
        if d['path']:
            st.write(f"**🛣️ Route:** {len(d['path'])} flights ({d['cost']:,.0f} km)")
        else:
            st.error("**🛣️ Route:** Unreachable by air")
        
        # --- NEW: LIVE MODEL ANALYTICS ---
        st.markdown("---")
        st.subheader("📊 Live Model Analytics")
        
        perc = d['perception']
        chart_col1, chart_col2 = st.columns(2)
        
        # Chart 1: Threat Probability Distribution (Bar Chart)
        with chart_col1:
            if perc['threat_probabilities']:
                probs = perc['threat_probabilities']
                # Sort them in logical order
                order = ['LOW', 'MEDIUM', 'HIGH', 'CRITICAL']
                ordered_probs = [probs.get(k, 0) for k in order]
                
                fig_threat = px.bar(
                    x=order, y=ordered_probs, 
                    labels={'x': 'Threat Level', 'y': 'Probability'},
                    title="Threat Model Confidence",
                    color=order,
                    color_discrete_map={'LOW':'#2ecc71', 'MEDIUM':'#f1c40f', 'HIGH':'#e67e22', 'CRITICAL':'#e74c3c'}
                )
                fig_threat.update_layout(height=250, margin=dict(l=20, r=20, t=40, b=20), showlegend=False)
                st.plotly_chart(fig_threat, use_container_width=True)
                
        # Chart 2: Tsunami Probability Gauge
        with chart_col2:
            tsunami_prob = perc['tsunami_probability'] * 100
            fig_tsu = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = tsunami_prob,
                title = {'text': "ML Tsunami Risk"},
                number = {'suffix': "%"},
                gauge = {
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 50], 'color': "lightgray"},
                        {'range': [50, 80], 'color': "gray"}
                    ],
                    'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 80}
                }
            ))
            fig_tsu.update_layout(height=250, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_tsu, use_container_width=True)
            
        st.markdown("---")
        st.write(f"**🚚 Transport:** {d['mode_str']}")
        st.write(f"**📦 Supplies:** {d['supplies']}")
        
        st.markdown("---")
        st.subheader("⚙️ Expert System (Forward Chaining)")
        
        # We use an expander so the raw trace logs don't overwhelm the UI
        with st.expander("View Inference Engine Trace Log"):
            # FIX: Use .get() to safely check for the key
            if d.get('trace_logs'):
                for step in d['trace_logs']:
                    st.caption(f"✓ {step}")
            else:
                st.caption("No rules fired for these parameters (or waiting for fresh simulation).")
                
        st.write("**Tactical Deployments Ordered:**")
        # FIX: Use .get() here as well
        if d.get('actionable_deployments'):
            for action in d['actionable_deployments']:
                st.markdown(f"- 🚨 **{action}**")
        else:
            st.markdown("- No specialized tactical deployments required.")
        
        # Clear Button
        if st.button("Reset Simulation"):
            st.session_state.processed_click = None
            st.session_state.final_map = None
            st.session_state.dashboard_data = None
            st.rerun()
            
        
            
    else:
        st.warning("Waiting for data... Click anywhere on the map to trigger a seismic event.")