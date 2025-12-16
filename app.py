
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

# =======================================================
# 1. LOAD ARTIFACTS
# =======================================================
try:
    # Load the trained model, scaler, and clustered data
    with open('kmeans_model.pkl', 'rb') as f:
        kmeans = pickle.load(f)
    
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    
    df_clustered = pd.read_csv('youtube_channels_clustered.csv')
    
    # Define features used for clustering (MUST MATCH THE TRAINING CODE)
    FEATURES = ['subscribers', 'video views', 'uploads', 'highest_yearly_earnings']
    K = kmeans.n_clusters
    
except FileNotFoundError as e:
    st.error(f"FATAL ERROR: Required file not found. Please ensure all three files are in the same folder as this app.py: {e}")
    st.stop()
except Exception as e:
    st.error(f"An error occurred loading model artifacts: {e}")
    st.stop()


# =======================================================
# 2. HELPER FUNCTIONS
# =======================================================

def get_cluster_profiles(df, features):
    """Calculates the mean of each feature for every cluster."""
    # Calculate the mean for the numerical features only
    profiles = df.groupby('cluster')[features].mean().reset_index()
    
    # Calculate the count/size of each cluster
    cluster_size = df['cluster'].value_counts().reset_index()
    cluster_size.columns = ['cluster', 'Count']
    profiles = pd.merge(profiles, cluster_size, on='cluster')

    # Apply a format for better display (Millions/Billions)
    profiles['subscribers'] = profiles['subscribers'].apply(lambda x: f'{x/1e6:,.1f} M')
    profiles['video views'] = profiles['video views'].apply(lambda x: f'{x/1e9:,.2f} B')
    profiles['highest_yearly_earnings'] = profiles['highest_yearly_earnings'].apply(lambda x: f'${x/1e6:,.1f} M')
    
    return profiles.rename(columns={'cluster': 'Archetype ID'})

def predict_new_channel(input_data):
    """Scales new input and predicts the cluster ID."""
    # Convert input data to the required 2D array format for the scaler
    new_data = np.array([list(input_data.values())])
    
    # Scale the input using the pre-trained scaler
    scaled_input = scaler.transform(new_data)
    
    # Predict the cluster
    prediction = kmeans.predict(scaled_input)
    return prediction[0]

# =======================================================
# 3. STREAMLIT APP LAYOUT
# =======================================================

st.set_page_config(layout="wide", page_title="YouTube Channel Profiler")

st.title("📺 YouTube Channel Success Profiler (K-Means Clustering)")

st.sidebar.header("Analyzer Settings")
selected_tab = st.sidebar.radio("Navigation", ["Archetype Profiles", "Channel Scout (Predict)"])

# --- TAB 1: ARCHETYPE PROFILES (Main Output) ---
if selected_tab == "Archetype Profiles":
    
    st.header(f"🧠 Discovered Channel Archetypes (K={K})")
    st.markdown("The K-Means model clustered top global channels into distinct success profiles based on their core metrics.")

    # Display Cluster Profiles
    cluster_profiles = get_cluster_profiles(df_clustered, FEATURES)
    
    st.subheader("📊 Cluster Mean Values (What defines each Archetype)")
    st.dataframe(cluster_profiles, hide_index=True)
    
    st.markdown("---")

    # Visualization: Scatter Plot
    st.subheader("📈 Visualization: Subscribers vs. Views (Log Scale)")
    st.markdown("Channels in the same color group share similar underlying success metrics.")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Use log scale for better visualization of highly skewed data
    sns.scatterplot(
        x=np.log1p(df_clustered['subscribers']), 
        y=np.log1p(df_clustered['video views']), 
        hue='cluster', 
        data=df_clustered, 
        palette='viridis', 
        legend='full',
        ax=ax
    )
    ax.set_title('Channel Clusters by Size and Reach (Log Scale)')
    ax.set_xlabel('Log(1 + Subscribers)')
    ax.set_ylabel('Log(1 + Video Views)')
    st.pyplot(fig)


# --- TAB 2: CHANNEL SCOUT (Prediction Tool) ---
elif selected_tab == "Channel Scout (Predict)":
    
    st.header("🔍 Predict Archetype for a New Channel")
    st.markdown("Enter hypothetical metrics for a channel to see which Success Archetype it falls into.")

    col1, col2 = st.columns(2)
    
    # Input fields for the user
    with col1:
        input_subs = st.number_input("Subscribers (Total)", min_value=0, value=5000000, step=100000)
        input_views = st.number_input("Video Views (Total)", min_value=0, value=500000000, step=10000000)

    with col2:
        input_uploads = st.number_input("Total Uploads", min_value=0, value=500, step=10)
        input_earnings = st.number_input("Highest Yearly Earnings (USD)", min_value=0, value=500000, step=10000)

    # Collect input data dictionary
    input_data = {
        'subscribers': input_subs,
        'video views': input_views,
        'uploads': input_uploads,
        'highest_yearly_earnings': input_earnings
    }

    if st.button("Predict Archetype", type="primary"):
        predicted_cluster = predict_new_channel(input_data)
        
        st.success(f"### Prediction Result: Archetype ID {predicted_cluster}")
        
        # Optionally show what that archetype means
        profiles = get_cluster_profiles(df_clustered, FEATURES)
        predicted_profile = profiles[profiles['Archetype ID'] == predicted_cluster]
        
        st.subheader("Characteristics of the Predicted Archetype")
        st.dataframe(predicted_profile, hide_index=True)
        
        st.info("This tool helps content managers quickly categorize channels for targeted strategy or investment.")
