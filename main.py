import streamlit as st
from google.cloud import aiplatform
import vertexai
from vertexai.generative_models import GenerativeModel
import fitdecode
import tcxparser
import json
import os
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import folium
from streamlit_folium import st_folium
from datetime import datetime, timedelta
from pprint import pprint
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure Vertex AI
vertexai.init(
    project=os.getenv("GOOGLE_CLOUD_PROJECT_ID"),  # Replace with your Google Cloud project ID
    location=os.getenv("GOOGLE_CLOUD_REGION")      # Replace with your preferred region
)
model = GenerativeModel(os.getenv("MODEL_NAME"))  # e.g., "gemini-pro"

# Page configuration
st.set_page_config(
    page_title="Fitness Data Analyzer",
    page_icon="🏃‍♂️",
    layout="wide",
    initial_sidebar_state="expanded"
)

def create_heart_rate_chart(hr_values, time_values=None):
    """Create an interactive heart rate chart"""
    if not hr_values or len(hr_values) == 0:
        return None
    
    df = pd.DataFrame({
        'Heart Rate': hr_values,
        'Time': range(len(hr_values)) if not time_values else time_values
    })
    
    fig = px.line(df, x='Time', y='Heart Rate', 
                  title='Heart Rate Over Time',
                  labels={'Time': 'Time (minutes)', 'Heart Rate': 'Heart Rate (bpm)'})
    
    # Add heart rate zones
    fig.add_hline(y=180, line_dash="dash", line_color="red", annotation_text="Max Zone")
    fig.add_hline(y=160, line_dash="dash", line_color="orange", annotation_text="Anaerobic")
    fig.add_hline(y=140, line_dash="dash", line_color="yellow", annotation_text="Aerobic")
    fig.add_hline(y=120, line_dash="dash", line_color="green", annotation_text="Fat Burn")
    
    fig.update_layout(height=400)
    return fig

def create_elevation_chart(altitude_points, distance_values=None):
    """Create an elevation profile chart"""
    if not altitude_points or len(altitude_points) == 0:
        return None
    
    df = pd.DataFrame({
        'Elevation': altitude_points,
        'Distance': range(len(altitude_points)) if not distance_values else distance_values
    })
    
    fig = px.area(df, x='Distance', y='Elevation',
                  title='Elevation Profile',
                  labels={'Distance': 'Distance (km)', 'Elevation': 'Elevation (m)'})
    
    fig.update_layout(height=300)
    return fig

def create_gps_map(position_values):
    """Create a GPS route map"""
    if not position_values or len(position_values) == 0:
        return None
    
    # Extract lat/lon from position values
    coords = []
    for pos in position_values:
        if isinstance(pos, (list, tuple)) and len(pos) >= 2:
            coords.append([pos[0], pos[1]])
    
    if not coords:
        return None
    
    # Create map centered on route
    center_lat = sum(coord[0] for coord in coords) / len(coords)
    center_lon = sum(coord[1] for coord in coords) / len(coords)
    
    m = folium.Map(location=[center_lat, center_lon], zoom_start=13)
    
    # Add route line
    folium.PolyLine(coords, color="red", weight=3, opacity=0.8).add_to(m)
    
    # Add start and end markers
    if len(coords) > 0:
        folium.Marker(coords[0], popup="Start", icon=folium.Icon(color="green")).add_to(m)
        folium.Marker(coords[-1], popup="End", icon=folium.Icon(color="red")).add_to(m)
    
    return m

def create_summary_metrics(workout_data_dict, filetype):
    """Create summary metrics cards"""
    metrics = {}

    if filetype == "FIT":
        if 'summary' in workout_data_dict:
            summary = workout_data_dict['summary']
            if 'total_timer_time' in summary:
                duration = int(float(summary['total_timer_time'])) if summary['total_timer_time'] else 0
                hours = duration // 3600
                minutes = (duration % 3600) // 60
                seconds = duration % 60
                metrics['Duration'] = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
            
            if 'total_distance' in summary:
                distance_km = float(summary['total_distance']) / 1000 if summary['total_distance'] else 0
                metrics['Distance'] = f"{distance_km:.2f} km"
            
            if 'total_calories' in summary:
                calories = int(float(summary['total_calories'])) if summary['total_calories'] else 0
                metrics['Calories'] = f"{calories} kcal"
            
            if 'avg_heart_rate' in summary and summary['avg_heart_rate']:
                avg_hr = int(float(summary['avg_heart_rate']))
                metrics['Avg Heart Rate'] = f"{avg_hr} bpm"
            if 'max_heart_rate' in summary and summary['max_heart_rate']:
                max_hr = int(float(summary['max_heart_rate']))
                metrics['Max Heart Rate'] = f"{max_hr} bpm"
            if 'min_heart_rate' in summary and summary['min_heart_rate']:
                min_hr = int(float(summary['min_heart_rate']))
                metrics['Min Heart Rate'] = f"{min_hr} bpm"
            
            if 'total_training_effect' in summary:
                tte = float(summary['total_training_effect'])
                metrics['Total Training Effect'] = f"{tte:.2f}"
            if 'total_anaerobic_training_effect' in summary:
                tate = float(summary['total_anaerobic_training_effect'])
                metrics['Total Anaerobic Training Effect'] = f"{tate:.2f}"
            if "training_load_peak" in summary:
                tlp = float(summary['training_load_peak'])
                metrics['Training Load Peak'] = f"{tlp:.2f}"
  
    elif filetype == "TCX":
        if 'duration' in workout_data_dict:
            duration = workout_data_dict['duration']
            hours = duration // 3600
            minutes = (duration % 3600) // 60
            seconds = duration % 60
            metrics['Duration'] = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        
        if 'distance' in workout_data_dict:
            distance_km = workout_data_dict['distance'] / 1000 if workout_data_dict['distance'] else 0
            metrics['Distance'] = f"{distance_km:.2f} km"
        
        if 'calories' in workout_data_dict:
            metrics['Calories'] = f"{workout_data_dict['calories']} kcal"
        
        if 'hr_avg' in workout_data_dict and workout_data_dict['hr_avg']:
            metrics['Avg Heart Rate'] = f"{workout_data_dict['hr_avg']} bpm"
        
        if 'avg_speed_kmh' in workout_data_dict:
            metrics['Avg Speed'] = f"{workout_data_dict['avg_speed_kmh']} km/h"
    
    return metrics

def parse_fit_file(file_path):
    """Parse FIT file and extract workout data"""
    try:
        print(f"Parsing FIT file: {file_path}")
        workout_data = {
            "type": "FIT",
            "records": [],
            "summary": {}
        }
        
        with fitdecode.FitReader(file_path) as fit:
            for frame in fit:
                if isinstance(frame, fitdecode.FitDataMessage):
                    if frame.name == 'record':
                        record = {}
                        for field in frame.fields:
                            record[field.name] = field.value
                        workout_data["records"].append(record)
                    elif frame.name == 'session':
                        for field in frame.fields:
                            workout_data["summary"][field.name] = field.value
        
        # Clean up temp file
        if os.path.exists(file_path):
            os.remove(file_path)
            
        return json.dumps(workout_data, default=str)
    except Exception as e:
        return json.dumps({"error": f"Failed to parse FIT file: {str(e)}"})

def parse_tcx_file(file_path):
    """Parse TCX file and extract workout data"""
    try:
        print(f"Parsing TCX file: {file_path}")
        tcx = tcxparser.TCXParser(file_path)
        # pprint(f"{dir(tcx)}")
        
        def format_pace(pace_seconds):
            if not pace_seconds:
                return None
            minutes = int(pace_seconds // 60)
            seconds = int(pace_seconds % 60)
            return f"{minutes}:{seconds:02d} /km"
        
        # Helper function to safely get attribute values
        def safe_get(obj, attr, default=None):
            try:
                value = getattr(obj, attr, default)
                # Check if it's a method and call it
                if callable(value):
                    try:
                        value = value()
                    except:
                        return default
                return value if value is not None else default
            except:
                return default
        
        workout_data = {
            "type": "TCX",
            "activity_type": safe_get(tcx, 'activity_type', 'Unknown'),
            "activity": safe_get(tcx, 'activity'),
            "activity_notes": safe_get(tcx, 'activity_notes', 'Activity'),
            "started_at": str(tcx.started_at) if safe_get(tcx, 'started_at') else None,
            "completed_at": str(tcx.completed_at) if safe_get(tcx, 'completed_at') else None,
            
            # Duration and Distance
            "duration": safe_get(tcx, 'duration', 0),
            "distance": safe_get(tcx, 'distance', 0),
            "distance_units": safe_get(tcx, 'distance_units', 'meters'),
            "distance_values": safe_get(tcx, 'distance_values', []),
            
            # Calories and Steps
            "calories": safe_get(tcx, 'calories', 0),
            "total_steps": safe_get(tcx, 'total_steps', 0),
            "steps_values": safe_get(tcx, 'steps_values', []),
            
            # Heart Rate Data (comprehensive)
            "hr_avg": safe_get(tcx, 'hr_avg'),
            "hr_max": safe_get(tcx, 'hr_max'),
            "hr_min": safe_get(tcx, 'hr_min'),
            "hr_values": safe_get(tcx, 'hr_values', []),
            "hr_time_in_zones": safe_get(tcx, 'hr_time_in_zones', []),
            "hr_percent_in_zones": safe_get(tcx, 'hr_percent_in_zones', []),
            
            # Altitude Data (comprehensive)
            "altitude_avg": safe_get(tcx, 'altitude_avg'),
            "altitude_max": safe_get(tcx, 'altitude_max'),
            "altitude_min": safe_get(tcx, 'altitude_min'),
            "altitude_points": safe_get(tcx, 'altitude_points', []),
            "ascent": safe_get(tcx, 'ascent'),
            "descent": safe_get(tcx, 'descent'),
            
            # Cadence Data (comprehensive)
            "cadence_avg": safe_get(tcx, 'cadence_avg'),
            "cadence_max": safe_get(tcx, 'cadence_max'),
            "cadence_values": safe_get(tcx, 'cadence_values', []),
            
            # Power Data (comprehensive)
            "power_avg": safe_get(tcx, 'power_avg'),
            "power_max": safe_get(tcx, 'power_max'),
            "power_values": safe_get(tcx, 'power_values', []),
            
            # Pace and Position Data
            "pace": safe_get(tcx, 'pace'),
            "latitude": safe_get(tcx, 'latitude'),
            "longitude": safe_get(tcx, 'longitude'),
            "position_values": safe_get(tcx, 'position_values', []),
            
            # Time series data (comprehensive)
            "time_values": [],
            "time_objects": safe_get(tcx, 'time_objects', []),
            "time_durations": safe_get(tcx, 'time_durations', []),
            "trackpoint_count": 0,
            
            # Calculate additional insights
            "duration_formatted": f"{safe_get(tcx, 'duration', 0) // 3600:02d}:{(safe_get(tcx, 'duration', 0) % 3600) // 60:02d}:{safe_get(tcx, 'duration', 0) % 60:02d}",
            "has_hr_data": bool(safe_get(tcx, 'hr_avg')),
            "has_gps_data": bool(safe_get(tcx, 'latitude') and safe_get(tcx, 'longitude')),
            "has_altitude_data": bool(safe_get(tcx, 'altitude_avg')),
            "has_cadence_data": bool(safe_get(tcx, 'cadence_avg')),
            "has_power_data": bool(safe_get(tcx, 'power_avg')),
            "has_steps_data": bool(safe_get(tcx, 'total_steps')),
            "workout_intensity": "Low" if safe_get(tcx, 'hr_avg', 0) < 120 else "Moderate" if safe_get(tcx, 'hr_avg', 0) < 150 else "High"
        }
        
        # Safely handle time_values which might be a method or property
        try:
            time_vals = safe_get(tcx, 'time_values', [])
            if time_vals and hasattr(time_vals, '__iter__'):
                workout_data["time_values"] = [str(t) for t in time_vals]
                workout_data["trackpoint_count"] = len(time_vals)
            else:
                workout_data["time_values"] = []
                workout_data["trackpoint_count"] = 0
        except Exception as e:
            print(f"Error processing time_values: {e}")
            workout_data["time_values"] = []
            workout_data["trackpoint_count"] = 0
        
        # Add comprehensive heart rate zones analysis if HR data is available
        if workout_data["has_hr_data"]:
            hr_avg = workout_data["hr_avg"]
            hr_max = workout_data["hr_max"]
            hr_min = workout_data["hr_min"]
            
        
        # Add pace analysis if available
        if safe_get(tcx, 'pace'):
            workout_data["pace_analysis"] = {
                "average_pace": safe_get(tcx, 'pace'),
                "pace_formatted": format_pace(safe_get(tcx, 'pace')) if safe_get(tcx, 'pace') else None
            }
        
        # Add calorie burn rate
        if workout_data["calories"] and workout_data["duration"]:
            calories_per_minute = workout_data["calories"] / (workout_data["duration"] / 60)
            workout_data["calorie_burn_rate"] = round(calories_per_minute, 1)
        
        # Add distance analysis if available
        if workout_data["distance"] and workout_data["duration"]:
            # Calculate average speed (assuming distance is in meters)
            avg_speed_ms = workout_data["distance"] / workout_data["duration"]  # m/s
            avg_speed_kmh = avg_speed_ms * 3.6  # km/h
            workout_data["avg_speed_kmh"] = round(avg_speed_kmh, 2)
            workout_data["avg_speed_mph"] = round(avg_speed_kmh * 0.621371, 2)
        
        # Clean up temp file
        if os.path.exists(file_path):
            os.remove(file_path)
            
        return json.dumps(workout_data, default=str)
    except Exception as e:
        print(f"Error parsing TCX file: {str(e)}")
        return json.dumps({"error": f"Failed to parse TCX file: {str(e)}"})

st.title("🏃‍♂️ Fitness Data Analyzer")
st.markdown("### Upload and analyze your workout files with AI-powered insights")

# Sidebar for app information and settings
with st.sidebar:
    st.header("📊 App Features")
    st.markdown("""
    - **File Support**: FIT & TCX files
    - **Visualizations**: Heart rate, elevation, GPS maps
    - **AI Analysis**: Powered by Google Vertex AI
    - **Export**: Download analysis reports
    """)
    
    st.header("🔧 Settings")
    show_raw_data = st.checkbox("Show raw data", value=False)
    auto_analyze = st.checkbox("Auto-analyze on upload", value=True)

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    # File upload with better styling
    st.markdown("#### 📁 Upload Workout File")
    uploaded_file = st.file_uploader(
        "Choose your fitness file", 
        type=['fit', 'tcx'],
        help="Upload FIT or TCX files from your fitness device (Garmin, Polar, etc.)",
        label_visibility="collapsed"
    )

with col2:
    if uploaded_file:
        st.markdown("#### 📋 File Info")
        st.info(f"""
        **File**: {uploaded_file.name}
        **Size**: {uploaded_file.size / 1024:.1f} KB
        **Type**: {uploaded_file.name.split('.')[-1].upper()}
        """)

# Process uploaded file
if uploaded_file:
    try:
        with st.spinner("🔄 Analyzing your workout..."):
            
            # Save temporarily and parse
            temp_filename = f"temp_{uploaded_file.name}"
            with open(temp_filename, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Parse based on file type
            if uploaded_file.name.endswith('.fit'):
                workout_data = parse_fit_file(temp_filename)
                filetype = "FIT"
            else:
                workout_data = parse_tcx_file(temp_filename)
                filetype = "TCX"
            
            # Parse JSON data for visualization
            workout_data_dict = json.loads(workout_data)
            
            if "error" in workout_data_dict:
                st.error(f"❌ {workout_data_dict['error']}")
            else:
                st.success("✅ Analysis Complete!")
                
                # Create tabs for different views
                tab1, tab2, tab3, tab4 = st.tabs(["📈 Overview", "💓 Heart Rate", "🗺️ Route", "🤖 AI Analysis"])
                
                with tab1:
                    # Summary metrics
                    st.markdown("#### 📊 Workout Summary")
                    metrics = create_summary_metrics(workout_data_dict, filetype)
                    
                    # Display metrics in columns only if we have metrics
                    if metrics:
                        metric_cols = st.columns(len(metrics))
                        for i, (key, value) in enumerate(metrics.items()):
                            with metric_cols[i]:
                                st.metric(key, value)
                    else:
                        st.info("No summary metrics available for this workout file.")
                    
                    # Additional details in expandable sections
                    col_left, col_right = st.columns(2)
                    
                    with col_left:
                        if workout_data_dict.get('altitude_points'):
                            st.markdown("#### 🏔️ Elevation Profile")
                            elevation_chart = create_elevation_chart(
                                workout_data_dict['altitude_points'],
                                workout_data_dict.get('distance_values')
                            )
                            if elevation_chart:
                                st.plotly_chart(elevation_chart, use_container_width=True)
                    
                    with col_right:
                        if workout_data_dict.get('activity_type'):
                            st.markdown("#### 🏃‍♂️ Activity Details")
                            activity_info = {
                                "Type": workout_data_dict.get('activity_type', 'Unknown'),
                                "Started": workout_data_dict.get('started_at', 'N/A'),
                                "Completed": workout_data_dict.get('completed_at', 'N/A'),
                                "Intensity": workout_data_dict.get('workout_intensity', 'Unknown')
                            }
                            for key, value in activity_info.items():
                                if value and value != 'N/A':
                                    st.write(f"**{key}**: {value}")
                
                with tab2:
                    if workout_data_dict.get('hr_values'):
                        st.markdown("#### 💓 Heart Rate Analysis")
                        hr_chart = create_heart_rate_chart(
                            workout_data_dict['hr_values'],
                            workout_data_dict.get('time_values')
                        )
                        if hr_chart:
                            st.plotly_chart(hr_chart, use_container_width=True)
                        
                        # HR Statistics
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            if workout_data_dict.get('hr_avg'):
                                st.metric("Average HR", f"{workout_data_dict['hr_avg']} bpm")
                        with col2:
                            if workout_data_dict.get('hr_max'):
                                st.metric("Maximum HR", f"{workout_data_dict['hr_max']} bpm")
                        with col3:
                            if workout_data_dict.get('hr_min'):
                                st.metric("Minimum HR", f"{workout_data_dict['hr_min']} bpm")
                    else:
                        st.info("No heart rate data available in this workout file.")
                
                with tab3:
                    if workout_data_dict.get('position_values'):
                        st.markdown("#### 🗺️ GPS Route")
                        gps_map = create_gps_map(workout_data_dict['position_values'])
                        if gps_map:
                            st_folium(gps_map, width=700, height=400)
                        else:
                            st.info("GPS data found but couldn't create map visualization.")
                    elif workout_data_dict.get('latitude') and workout_data_dict.get('longitude'):
                        st.markdown("#### 📍 Workout Location")
                        simple_map = folium.Map(
                            location=[workout_data_dict['latitude'], workout_data_dict['longitude']], 
                            zoom_start=15
                        )
                        folium.Marker(
                            [workout_data_dict['latitude'], workout_data_dict['longitude']],
                            popup="Workout Location"
                        ).add_to(simple_map)
                        st_folium(simple_map, width=700, height=400)
                    else:
                        st.info("No GPS data available in this workout file.")
                
                with tab4:
                    if auto_analyze:
                        st.markdown("#### 🤖 AI-Powered Analysis")
                        with st.spinner("Getting AI insights..."):
                            # Enhanced AI analysis prompt
                            prompt = f"""
                            Analyze this comprehensive workout data and provide detailed insights:
                            {workout_data}
                            
                            Please provide analysis in the following structured format:
                            
                            ## 🏃‍♂️ **Workout Overview**
                            [Brief summary of the workout type, duration, and intensity]
                            
                            ## 📊 **Performance Metrics**
                            [Analysis of key performance indicators like pace, heart rate, power, etc.]
                            
                            ## 💪 **Training Zones Analysis**
                            [Heart rate zones, intensity distribution, training effect]
                            
                            ## 🎯 **Recommendations**
                            [Specific actionable recommendations for improvement]
                            
                            ## 🔄 **Recovery Suggestions**
                            [Recovery time, nutrition, hydration recommendations]
                            
                            Keep the analysis practical, motivating, and specific to the data provided.
                            """
                            
                            try:
                                response = model.generate_content(prompt)
                                st.markdown(response.text)
                            except Exception as e:
                                st.error(f"AI Analysis failed: {str(e)}")
                                st.info("Please check your Vertex AI configuration and try again.")
                    else:
                        st.info("Enable 'Auto-analyze on upload' in the sidebar or click below to get AI insights.")
                        if st.button("🤖 Get AI Analysis"):
                            # Same AI analysis code as above
                            pass
                
                # Raw data section
                if show_raw_data:
                    with st.expander("🔍 View Raw Data", expanded=False):
                        st.json(workout_data_dict)
                
                # Store workout data in session for chat
                st.session_state.current_workout = workout_data
                
    except Exception as e:
        st.error(f"❌ Error processing file: {str(e)}")
        st.info("Please ensure your file is a valid FIT or TCX file and try again.")

# Enhanced Chat interface for follow-up questions
st.markdown("---")
st.markdown("#### 💬 Chat with AI about your workout")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history with better styling
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input with context awareness
if prompt := st.chat_input("Ask questions about your workout data..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate response with context
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                # Include workout context if available
                context = ""
                if hasattr(st.session_state, 'current_workout'):
                    context = f"Current workout data: {st.session_state.current_workout}\n\n"
                
                full_prompt = f"""
                {context}
                User question: {prompt}
                
                Please provide a helpful, specific answer about the workout data. 
                If no workout data is available, let the user know they need to upload a file first.
                Keep responses conversational but informative.
                """
                
                response = model.generate_content(full_prompt)
                st.markdown(response.text)
                
                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": response.text})
                
            except Exception as e:
                error_msg = f"Sorry, I encountered an error: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})

# Chat management
col1, col2, col3 = st.columns([1, 1, 2])
with col1:
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()

with col2:
    if st.button("💾 Export Chat"):
        if st.session_state.messages:
            chat_export = "\n\n".join([f"{msg['role'].title()}: {msg['content']}" for msg in st.session_state.messages])
            st.download_button(
                label="Download Chat History",
                data=chat_export,
                file_name=f"workout_chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain"
            )

# Footer with additional information
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>🏃‍♂️ <strong>Fitness Data Analyzer</strong> | Powered by Google Vertex AI</p>
    <p><small>Supports FIT and TCX files from Garmin, Polar, Suunto, and other fitness devices</small></p>
</div>
""", unsafe_allow_html=True)
