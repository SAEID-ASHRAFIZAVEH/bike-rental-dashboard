"""
Washington D.C. Bike Rental Dashboard
Interactive Dashboard for Bike Rental Analysis (2011-2012)
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Page Configuration
st.set_page_config(
    page_title="DC Bike Rental Dashboard",
    page_icon="🚲",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #374151;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #F3F4F6;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .stSelectbox, .stSlider, .stRadio {
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load and preprocess the dataset"""
    df = pd.read_csv('data/train.csv')
    
    # Convert datetime
    df['datetime'] = pd.to_datetime(df['datetime'])
    
    # Create time-based features
    df['year'] = df['datetime'].dt.year
    df['month'] = df['datetime'].dt.month
    df['day_of_week'] = df['datetime'].dt.dayofweek
    df['hour'] = df['datetime'].dt.hour
    df['day'] = df['datetime'].dt.day
    
    # Rename seasons
    season_map = {1: 'Spring', 2: 'Summer', 3: 'Fall', 4: 'Winter'}
    df['season'] = df['season'].map(season_map)
    
    # Create day period
    conditions = [
        (df['hour'] >= 0) & (df['hour'] < 6),
        (df['hour'] >= 6) & (df['hour'] < 12),
        (df['hour'] >= 12) & (df['hour'] < 18),
        (df['hour'] >= 18) & (df['hour'] <= 23)
    ]
    choices = ['Night', 'Morning', 'Afternoon', 'Evening']
    df['day_period'] = np.select(conditions, choices, default='Night')
    
    # Weather descriptions
    weather_map = {
        1: 'Clear/Partly Cloudy',
        2: 'Mist/Cloudy',
        3: 'Light Snow/Rain',
        4: 'Heavy Rain/Snow'
    }
    df['weather_desc'] = df['weather'].map(weather_map)
    
    return df

def main():
    # Title and Description
    st.markdown('<h1 class="main-header">🚲 Washington D.C. Bike Rental Dashboard</h1>', unsafe_allow_html=True)
    st.markdown("""
    **Interactive analysis of Capital Bikeshare rentals (2011-2012)**
    Explore how weather, time, and user patterns affect bike rental demand.
    """)
    
    # Load data
    df = load_data()
    
    # Sidebar - Interactive Widgets
    st.sidebar.header("🎛️ Dashboard Controls")
    
    # Widget 1: Year Selection
    selected_years = st.sidebar.multiselect(
        "Select Year(s)",
        options=sorted(df['year'].unique()),
        default=sorted(df['year'].unique())
    )
    
    # Widget 2: Season Filter
    selected_seasons = st.sidebar.multiselect(
        "Select Season(s)",
        options=['Spring', 'Summer', 'Fall', 'Winter'],
        default=['Spring', 'Summer', 'Fall', 'Winter']
    )
    
    # Widget 3: Day Type Filter
    day_type = st.sidebar.radio(
        "Day Type",
        options=['All', 'Working Days', 'Non-Working Days']
    )
    
    # Widget 4: Temperature Range
    temp_range = st.sidebar.slider(
        "Temperature Range (°C)",
        min_value=float(df['temp'].min()),
        max_value=float(df['temp'].max()),
        value=(float(df['temp'].min()), float(df['temp'].max()))
    )
    
    # Widget 5: Hour Range
    hour_range = st.sidebar.slider(
        "Hour of Day",
        min_value=0,
        max_value=23,
        value=(7, 19)
    )
    
    # Filter data based on widgets
    filtered_df = df.copy()
    
    if selected_years:
        filtered_df = filtered_df[filtered_df['year'].isin(selected_years)]
    
    if selected_seasons:
        filtered_df = filtered_df[filtered_df['season'].isin(selected_seasons)]
    
    if day_type == 'Working Days':
        filtered_df = filtered_df[filtered_df['workingday'] == 1]
    elif day_type == 'Non-Working Days':
        filtered_df = filtered_df[filtered_df['workingday'] == 0]
    
    filtered_df = filtered_df[
        (filtered_df['temp'] >= temp_range[0]) & 
        (filtered_df['temp'] <= temp_range[1])
    ]
    
    filtered_df = filtered_df[
        (filtered_df['hour'] >= hour_range[0]) & 
        (filtered_df['hour'] <= hour_range[1])
    ]
    
    # Key Metrics Row
    st.markdown('<h2 class="sub-header">📊 Key Metrics</h2>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_rentals = filtered_df['count'].sum()
        st.metric("Total Rentals", f"{total_rentals:,.0f}")
    
    with col2:
        avg_daily = filtered_df.groupby('datetime').agg({'count': 'sum'}).mean()[0]
        st.metric("Avg Daily Rentals", f"{avg_daily:.0f}")
    
    with col3:
        registered_pct = (filtered_df['registered'].sum() / total_rentals * 100)
        st.metric("Registered Users", f"{registered_pct:.1f}%")
    
    with col4:
        peak_hour = filtered_df.groupby('hour')['count'].mean().idxmax()
        st.metric("Peak Hour", f"{peak_hour}:00")
    
    # Main Visualizations
    
    # Plot 1: Rentals by Hour with Confidence Interval
    st.markdown('<h2 class="sub-header">🕒 Hourly Rental Patterns</h2>', unsafe_allow_html=True)
    
    fig1 = go.Figure()
    
    # Calculate mean and confidence interval
    hourly_stats = filtered_df.groupby('hour').agg({
        'count': ['mean', 'std', 'count']
    }).reset_index()
    
    hourly_stats.columns = ['hour', 'mean', 'std', 'n']
    hourly_stats['ci'] = 1.96 * hourly_stats['std'] / np.sqrt(hourly_stats['n'])
    
    fig1.add_trace(go.Scatter(
        x=hourly_stats['hour'],
        y=hourly_stats['mean'] + hourly_stats['ci'],
        mode='lines',
        line=dict(width=0),
        showlegend=False,
        name='Upper CI'
    ))
    
    fig1.add_trace(go.Scatter(
        x=hourly_stats['hour'],
        y=hourly_stats['mean'] - hourly_stats['ci'],
        mode='lines',
        line=dict(width=0),
        fillcolor='rgba(68, 68, 68, 0.3)',
        fill='tonexty',
        showlegend=False,
        name='Lower CI'
    ))
    
    fig1.add_trace(go.Scatter(
        x=hourly_stats['hour'],
        y=hourly_stats['mean'],
        mode='lines+markers',
        line=dict(color='#1E3A8A', width=3),
        name='Mean Rentals'
    ))
    
    fig1.update_layout(
        title='Hourly Rentals with 95% Confidence Interval',
        xaxis_title='Hour of Day',
        yaxis_title='Average Rentals',
        hovermode='x unified',
        height=400
    )
    
    st.plotly_chart(fig1, use_container_width=True)
    
    # Plot 2: Rentals by Season and Weather
    st.markdown('<h2 class="sub-header">🌦️ Season & Weather Analysis</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Seasonal patterns
        season_avg = filtered_df.groupby('season')['count'].mean().reset_index()
        fig2a = px.bar(
            season_avg,
            x='season',
            y='count',
            title='Average Rentals by Season',
            color='season',
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        fig2a.update_layout(height=400)
        st.plotly_chart(fig2a, use_container_width=True)
    
    with col2:
        # Weather impact
        weather_avg = filtered_df.groupby('weather_desc')['count'].mean().reset_index()
        fig2b = px.bar(
            weather_avg,
            x='weather_desc',
            y='count',
            title='Average Rentals by Weather Condition',
            color='weather_desc',
            color_discrete_sequence=px.colors.sequential.Blues_r
        )
        fig2b.update_layout(height=400, xaxis_title='Weather Condition')
        st.plotly_chart(fig2b, use_container_width=True)
    
    # Plot 3: User Type Analysis
    st.markdown('<h2 class="sub-header">👥 User Type Comparison</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Registered vs Casual by Day Type
        user_by_daytype = filtered_df.groupby('workingday')[['registered', 'casual']].mean().reset_index()
        user_by_daytype['workingday'] = user_by_daytype['workingday'].map({0: 'Non-Working', 1: 'Working'})
        
        fig3a = go.Figure(data=[
            go.Bar(name='Registered', x=user_by_daytype['workingday'], y=user_by_daytype['registered']),
            go.Bar(name='Casual', x=user_by_daytype['workingday'], y=user_by_daytype['casual'])
        ])
        
        fig3a.update_layout(
            title='User Types: Working vs Non-Working Days',
            barmode='group',
            height=400,
            xaxis_title='Day Type',
            yaxis_title='Average Rentals'
        )
        st.plotly_chart(fig3a, use_container_width=True)
    
    with col2:
        # Monthly trends for both user types
        monthly_users = filtered_df.groupby('month')[['registered', 'casual']].mean().reset_index()
        fig3b = px.line(
            monthly_users,
            x='month',
            y=['registered', 'casual'],
            title='Monthly Rental Trends by User Type',
            markers=True
        )
        fig3b.update_layout(
            height=400,
            xaxis_title='Month',
            yaxis_title='Average Rentals',
            xaxis=dict(tickmode='array', tickvals=list(range(1, 13)))
        )
        st.plotly_chart(fig3b, use_container_width=True)
    
    # Plot 4: Correlation Heatmap
    st.markdown('<h2 class="sub-header">📈 Correlation Analysis</h2>', unsafe_allow_html=True)
    
    # Select only numerical columns for correlation
    numerical_cols = ['temp', 'atemp', 'humidity', 'windspeed', 'casual', 'registered', 'count']
    corr_matrix = filtered_df[numerical_cols].corr()
    
    fig4 = px.imshow(
        corr_matrix,
        text_auto='.2f',
        aspect='auto',
        color_continuous_scale='RdBu',
        title='Correlation Matrix of Numerical Variables'
    )
    fig4.update_layout(height=500)
    st.plotly_chart(fig4, use_container_width=True)
    
    # Plot 5: Day Period Analysis
    st.markdown('<h2 class="sub-header">🌙 Day Period Analysis</h2>', unsafe_allow_html=True)
    
    day_period_analysis = filtered_df.groupby(['day_period', 'workingday'])['count'].mean().reset_index()
    day_period_analysis['workingday'] = day_period_analysis['workingday'].map({0: 'Non-Working', 1: 'Working'})
    
    fig5 = px.bar(
        day_period_analysis,
        x='day_period',
        y='count',
        color='workingday',
        barmode='group',
        title='Average Rentals by Day Period (Working vs Non-Working Days)',
        category_orders={'day_period': ['Night', 'Morning', 'Afternoon', 'Evening']}
    )
    fig5.update_layout(height=400)
    st.plotly_chart(fig5, use_container_width=True)
    
    # Data Summary Table
    with st.expander("📋 View Filtered Data Summary"):
        st.dataframe(filtered_df.describe(), use_container_width=True)
    
    # Download filtered data
    csv = filtered_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Filtered Data as CSV",
        data=csv,
        file_name="filtered_bike_data.csv",
        mime="text/csv"
    )
    
    # Footer
    st.markdown("---")
    st.markdown("""
    **Dashboard Features:**
    - 5+ Interactive Visualizations
    - 5 Interactive Widgets (Year, Season, Day Type, Temperature, Hour Range)
    - Real-time Data Filtering
    - 95% Confidence Intervals
    - Correlation Analysis
    - Data Export Capability
    """)

if __name__ == "__main__":
    main()