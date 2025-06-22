# app.py
# FINAL VERSION with Live Performance, Robust Map, and UI Fixes
# To run this app:
# 1. Ensure Python 3.11 is installed.
# 2. Place 'delivery_model.pkl' in the same directory.
# 3. Create and activate a virtual environment with Python 3.11:
#    py -3.11 -m venv venv
#    venv\Scripts\activate
# 4. Install libraries:
#    pip install streamlit scikit-learn==1.3.2 joblib==1.4.2 xgboost==2.0.3 pandas==2.2.2 numpy==1.26.4 ortools==9.9.3963 openpyxl pydeck altair requests
# 5. Run the app:
#    streamlit run app.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pydeck as pdk
import altair as alt
import requests
from datetime import datetime
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
from sklearn.metrics import mean_absolute_error, r2_score
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# --- Page Configuration ---
st.set_page_config(page_title="Logistics Intelligence Dashboard", page_icon="🚀", layout="wide")

# --- Custom CSS ---
st.markdown("""
<style>
    .stApp { background-color: #0E1117; color: #FAFAFA; }
    h1 { color: #FFFFFF; text-align: center; }
    h2, h3 { color: #00A6FF; }
</style>
""", unsafe_allow_html=True)


# --- Helper & API Functions ---
def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371
    try:
        lat1_rad, lon1_rad, lat2_rad, lon2_rad = map(np.radians, [lat1, lon1, lat2, lon2])
        dlon = lon2_rad - lon1_rad
        dlat = lat2_rad - lat1_rad
        a = np.sin(dlat / 2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2)**2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
        return R * c
    except (ValueError, TypeError): return 0

def feature_engineer(df):
    df_processed = df.copy()
    df_processed['distance_km'] = df_processed.apply(lambda r: haversine_distance(r['origin_lat'], r['origin_lon'], r['dest_lat'], r['dest_lon']), axis=1)
    df_processed['order_placed_time'] = pd.to_datetime(df_processed['order_placed_time'])
    df_processed['order_hour'] = df_processed['order_placed_time'].dt.hour
    df_processed['order_day_of_week'] = df_processed['order_placed_time'].dt.dayofweek
    return df_processed

def get_optimized_route(locations):
    if not locations or len(locations) <= 1: return [], 0
    data = {'locations': locations, 'num_vehicles': 1, 'depot': 0}
    manager = pywrapcp.RoutingIndexManager(len(data['locations']), data['num_vehicles'], data['depot'])
    routing = pywrapcp.RoutingModel(manager)
    def distance_callback(f, t):
        n1, n2 = manager.IndexToNode(f), manager.IndexToNode(t)
        return int(haversine_distance(data['locations'][n1][0], data['locations'][n1][1], data['locations'][n2][0], data['locations'][n2][1]) * 1000)
    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
    solution = routing.SolveWithParameters(search_parameters)
    if solution:
        index, route_indices = routing.Start(0), []
        while not routing.IsEnd(index):
            route_indices.append(manager.IndexToNode(index))
            index = solution.Value(routing.NextVar(index))
        route_indices.append(manager.IndexToNode(index))
        return route_indices, solution.ObjectiveValue() / 1000
    else: return None, 0

@st.cache_data
def get_mapbox_route_in_chunks(coordinates, api_key, chunk_size=20):
    all_route_coords = []
    for i in range(0, len(coordinates), chunk_size - 1):
        chunk = coordinates[i:i + chunk_size]
        if len(chunk) < 2: continue
        coords_str = ";".join([f"{lon},{lat}" for lat, lon in chunk])
        url = f"https://api.mapbox.com/directions/v5/mapbox/driving/{coords_str}"
        params = {'geometries': 'geojson', 'access_token': api_key}
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            if data['routes']:
                all_route_coords.extend(data['routes'][0]['geometry']['coordinates'])
        except requests.exceptions.RequestException:
            st.warning("Could not fetch real road route from Mapbox. Displaying straight lines.", icon="⚠️")
            return None
    return all_route_coords

@st.cache_resource
def load_model():
    try: return joblib.load('delivery_model.pkl')
    except FileNotFoundError: return None

# --- Main App UI ---
st.title("🚀 Logistics Intelligence Dashboard")

with st.sidebar:
    st.header("Configuration")
    mapbox_api_key = st.text_input("Enter Mapbox API Key", type="password", help="Get a free key from mapbox.com")
    with st.expander("Historical Model Performance"):
        st.info("**MAE:** 2.76 minutes\n\n**R² Score:** 0.96", icon="🎯")

model = load_model()
if model is None:
    st.error("🚨 Model file 'delivery_model.pkl' not found.", icon="🔥")
    st.stop()

uploaded_file = st.file_uploader("Upload delivery manifest (must include 'actual_delivery_minutes' for live performance)", type=['csv', 'xlsx'])

if uploaded_file is not None:
    try:
        df_original = pd.read_excel(uploaded_file) if uploaded_file.name.endswith('.xlsx') else pd.read_csv(uploaded_file)
        
        with st.spinner('Analyzing routes and predicting times...'):
            df_featured = feature_engineer(df_original.copy())
            df_featured['predicted_duration_mins'] = np.round(model.predict(df_featured), 2)
            depot_location = (df_featured.iloc[0]['origin_lat'], df_featured.iloc[0]['origin_lon'])
            all_locations = [depot_location] + list(zip(df_featured['dest_lat'], df_featured['dest_lon']))
            original_total_distance = sum(haversine_distance(all_locations[i][0], all_locations[i][1], all_locations[i+1][0], all_locations[i+1][1]) for i in range(len(all_locations) - 1)) + haversine_distance(all_locations[-1][0], all_locations[-1][1], depot_location[0], depot_location[1])
            optimized_route_indices, optimized_total_distance = get_optimized_route(all_locations)
        
        st.success("Analysis complete!", icon="✅")
        
        if optimized_route_indices:
            # --- LIVE PERFORMANCE SECTION ---
            if 'actual_delivery_minutes' in df_featured.columns:
                with st.container(border=True):
                    st.header("Live Performance on Uploaded Data", divider='blue')
                    live_mae = mean_absolute_error(df_featured['actual_delivery_minutes'], df_featured['predicted_duration_mins'])
                    live_r2 = r2_score(df_featured['actual_delivery_minutes'], df_featured['predicted_duration_mins'])
                    perf_cols = st.columns(2)
                    perf_cols[0].metric("Live Mean Absolute Error", f"{live_mae:.2f} min")
                    perf_cols[1].metric("Live R² Score", f"{live_r2:.2f}")
                    
                    # Add a chart comparing actual vs predicted
                    chart_data = df_featured[['actual_delivery_minutes', 'predicted_duration_mins']].copy()
                    chart_data['error'] = chart_data['predicted_duration_mins'] - chart_data['actual_delivery_minutes']
                    
                    scatter_chart = alt.Chart(chart_data).mark_circle(size=60, opacity=0.7).encode(
                        x=alt.X('actual_delivery_minutes', title='Actual Time (min)'),
                        y=alt.Y('predicted_duration_mins', title='Predicted Time (min)'),
                        tooltip=['actual_delivery_minutes', 'predicted_duration_mins', 'error']
                    ).interactive()
                    
                    line_chart = alt.Chart(pd.DataFrame({'x': [0, chart_data.max().max()], 'y': [0, chart_data.max().max()]})).mark_line(color='red', strokeDash=[3,3]).encode(x='x', y='y')
                    
                    st.altair_chart(scatter_chart + line_chart, use_container_width=True)
                    st.caption("Points closer to the red dashed line represent more accurate predictions.")

            # --- KPI Summary ---
            st.header("📈 Route Optimization Summary", divider='rainbow')
            distance_saved = original_total_distance - optimized_total_distance
            improvement_percent = (distance_saved / original_total_distance) * 100 if original_total_distance > 0 else 0
            final_plan = df_featured.iloc[[idx - 1 for idx in optimized_route_indices if idx != 0]].copy()
            
            kpi_cols = st.columns(3)
            kpi_cols[0].metric("Original Distance", f"{original_total_distance:.1f} km")
            kpi_cols[1].metric("Optimized Distance", f"{optimized_total_distance:.1f} km", delta=f"{-distance_saved:.1f} km")
            kpi_cols[2].metric("Efficiency Gain", f"{improvement_percent:.1f} %")

            # --- Dashboard ---
            map_col, chart_col = st.columns([3, 2])
            with map_col:
                st.subheader("Route Comparison Map")
                # Prepare coordinates for Mapbox API
                original_path_coords = all_locations + [all_locations[0]]
                optimized_path_coords = [all_locations[i] for i in optimized_route_indices]
                
                # Default to straight lines
                original_route_geom = [{'path': [[lon, lat] for lat, lon in original_path_coords]}]
                optimized_route_geom = [{'path': [[lon, lat] for lat, lon in optimized_path_coords]}]

                if mapbox_api_key:
                    with st.spinner("Fetching real road routes from Mapbox..."):
                        mapbox_optimized = get_mapbox_route_in_chunks(optimized_path_coords, mapbox_api_key)
                        if mapbox_optimized: optimized_route_geom = [{'path': mapbox_optimized}]
                        mapbox_original = get_mapbox_route_in_chunks(original_path_coords, mapbox_api_key)
                        if mapbox_original: original_route_geom = [{'path': mapbox_original}]
                else:
                    st.info("Enter a Mapbox API Key in the sidebar to see routes on real roads.", icon="🔑")

                st.pydeck_chart(pdk.Deck(
                    map_style='mapbox://styles/mapbox/dark-v10',
                    initial_view_state=pdk.ViewState(latitude=df_featured['dest_lat'].mean(), longitude=df_featured['dest_lon'].mean(), zoom=10, pitch=45),
                    layers=[
                        pdk.Layer('PathLayer', data=pd.DataFrame(original_route_geom), get_path='path', get_width=5, get_color=[255, 80, 80, 90], width_min_pixels=3),
                        pdk.Layer('PathLayer', data=pd.DataFrame(optimized_route_geom), get_path='path', get_width=8, get_color=[0, 200, 0, 200], width_min_pixels=4),
                        pdk.Layer('ScatterplotLayer', data=df_featured, get_position='[dest_lon, dest_lat]', get_color='[255, 140, 0, 200]', get_radius=150),
                        pdk.Layer('ScatterplotLayer', data=pd.DataFrame([{'lat': depot_location[0], 'lon': depot_location[1]}]), get_position='[lon, lat]', get_color='[0, 166, 255, 255]', get_radius=250)
                    ]
                ))

            with chart_col:
                st.subheader("Delivery Insights")
                with st.container(border=True):
                    st.altair_chart(alt.Chart(df_featured).mark_bar(color='#00A6FF').encode(x=alt.X('order_hour:O', title='Hour'), y=alt.Y('count():Q', title='Deliveries')).configure_axis(grid=False).configure_view(strokeOpacity=0), use_container_width=True)
                with st.container(border=True):
                    st.altair_chart(alt.Chart(df_featured).mark_arc(innerRadius=60).encode(theta=alt.Theta(field="vehicle_type", type="nominal"), color=alt.Color(field="vehicle_type", type="nominal", legend=None), tooltip=['vehicle_type', 'count()']), use_container_width=True)

            st.header("📦 Delivery Plans", divider='rainbow')
            final_plan['stop_number'] = range(1, len(final_plan) + 1)
            st.dataframe(final_plan[['stop_number', 'delivery_id', 'dest_lat', 'dest_lon', 'vehicle_type', 'predicted_duration_mins']], use_container_width=True)
            st.download_button(label="📥 Download Optimized Plan", data=final_plan.to_csv(index=False).encode('utf-8'), file_name='optimized_delivery_plan.csv', mime='text/csv')
        else:
            st.error("Could not find an optimized route.", icon="🤷")
    except Exception as e:
        st.error(f"An error occurred: {e}", icon="🔥")
        st.exception(e)

