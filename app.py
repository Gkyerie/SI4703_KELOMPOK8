import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor, plot_tree, export_text # Import plot_tree and export_text
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, r2_score, mean_squared_error
import os

st.set_page_config(
    page_title="Analisis Pengaruh Pendidikan terhadap Tingkat Pengangguran",
    page_icon="📊",
    layout="wide"
)

st.title("Analisis Pengaruh Tingkat Pendidikan terhadap Tingkat Pengangguran di Indonesia")

# --- Data Loading and Preparation ---
st.subheader("Data Loading and Preparation")

uploaded_file = st.file_uploader("Upload dataset CSV", type="csv")

df = None # Initialize df to None

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)

        # Data Preparation Steps from your notebook
        tahun_pendidikan = {
            'Tidak/belum pernah sekolah': 0,
            'Tidak/belum tamat SD': 3,
            'SD': 6,
            'SLTP': 9,
            'SLTA Umum/SMU': 12,
            'SLTA Kejuruan/SMK': 12,
            'Akademi/Diploma': 14,
            'Universitas': 16
        }

        # Ensure necessary columns exist before accessing them
        required_edu_cols = list(tahun_pendidikan.keys()) + ['Total']
        if not all(col in df.columns for col in required_edu_cols):
            st.error("Error: Dataset is missing required education level columns or 'Total'.")
            df = None # Invalidate df if columns are missing
        else:
            df['Total_Pendidikan_Terpenuhi'] = 0
            for level, years in tahun_pendidikan.items():
                df['Total_Pendidikan_Terpenuhi'] += df[level] * years

            # Handle potential division by zero if 'Total' is zero
            df['Pendidikan_RataRata'] = df.apply(
                lambda row: row['Total_Pendidikan_Terpenuhi'] / row['Total'] if row['Total'] > 0 else 0,
                axis=1
            )

            # Simulate 'Tingkat_Pengangguran' as done in your notebook
            # NOTE: In a real scenario, this column would be in your dataset.
            # Since you simulated it, we'll keep the simulation here for demonstration.
            np.random.seed(42)
            df['Tingkat_Pengangguran'] = 12 * np.exp(-0.2 * df['Pendidikan_RataRata']) + np.random.normal(0, 0.3, size=len(df))
            df['Tingkat_Pengangguran'] = df['Tingkat_Pengangguran'].clip(lower=1) # minimal 1%


            st.success("Data loaded and prepared successfully!")
            if 'Periode' in df.columns:
                 st.dataframe(df[['Periode', 'Pendidikan_RataRata', 'Tingkat_Pengangguran']].head())
            else:
                 st.dataframe(df[['Pendidikan_RataRata', 'Tingkat_Pengangguran']].head())


    except Exception as e:
        st.error(f"Error loading or preparing data: {e}")
        df = None # Invalidate df on any loading/preparation error

# --- Model Loading ---
st.subheader("Model Loading")

# Define the expected model filenames
linear_regression_model_file = 'linear_regression_model.pkl'
decision_tree_model_file = 'decision_tree_model.pkl'
kmeans_model_file = 'kmeans_model.pkl'
scaler_file = 'scaler.pkl' # Assuming you saved the scaler too

models = {}
scaler = None
model_loading_success = True

# Check if model files exist before attempting to load
if os.path.exists(linear_regression_model_file) and \
   os.path.exists(decision_tree_model_file) and \
   os.path.exists(kmeans_model_file) and \
   os.path.exists(scaler_file):

    try:
        # Load Linear Regression model
        with open(linear_regression_model_file, 'rb') as f:
            models['Linear Regression'] = pickle.load(f)
        st.success(f"Linear Regression model loaded from '{linear_regression_model_file}'")

        # Load Decision Tree model
        with open(decision_tree_model_file, 'rb') as f:
            models['Decision Tree'] = pickle.load(f)
        st.success(f"Decision Tree model loaded from '{decision_tree_model_file}'")

        # Load KMeans model
        with open(kmeans_model_file, 'rb') as f:
            models['KMeans Clustering'] = pickle.load(f)
        st.success(f"KMeans Clustering model loaded from '{kmeans_model_file}'")

        # Load Scaler
        with open(scaler_file, 'rb') as f:
            scaler = pickle.load(f)
        st.success(f"Scaler loaded from '{scaler_file}'")

    except Exception as e:
        st.error(f"Error loading models: {e}")
        models = {} # Clear models if loading fails
        scaler = None
        model_loading_success = False
else:
    st.warning("Model files not found. Please ensure 'linear_regression_model.pkl', 'decision_tree_model.pkl', 'kmeans_model.pkl', and 'scaler.pkl' are in the same directory.")
    model_loading_success = False


# --- Model Application and Visualization ---
# Proceed only if data is loaded and models are loaded successfully
if df is not None and models and scaler is not None and model_loading_success:
    st.subheader("Model Results and Visualizations")

    # --- K-Means Clustering with Button ---
    if 'KMeans Clustering' in models:
        st.write("#### K-Means Clustering")

        # Allow user to select the number of clusters
        n_clusters_input = st.slider(
            "Select the number of clusters for K-Means:",
            min_value=2,
            max_value=10,
            value=models['KMeans Clustering'].n_clusters, # Default to loaded model's clusters
            step=1
        )

        # Add a button to trigger clustering
        if st.button(f'Perform K-Means Clustering with {n_clusters_input} clusters'):
             try:
                features_for_clustering = ['Pendidikan_RataRata', 'Tingkat_Pengangguran']
                if not all(col in df.columns for col in features_for_clustering):
                     st.error(f"Clustering features {features_for_clustering} not found in data.")
                else:
                    X_clustering = df[features_for_clustering]
                    X_clustering_scaled = scaler.transform(X_clustering) # Use the loaded scaler

                    # Perform K-Means clustering with the selected number of clusters
                    kmeans_dynamic = KMeans(n_clusters=n_clusters_input, random_state=42, n_init=10) # Specify n_init
                    df['cluster'] = kmeans_dynamic.fit_predict(X_clustering_scaled)

                    cluster_summary = df.groupby('cluster').agg({
                        'Pendidikan_RataRata': 'mean',
                        'Tingkat_Pengangguran': 'mean',
                         'Periode': 'count' if 'Periode' in df.columns else ('Pendidikan_RataRata', 'count') # Count using any column if Periode is missing
                    }).round(2)
                    cluster_summary.rename(columns={cluster_summary.columns[-1]: 'Jumlah Data'}, inplace=True) # Rename the last column to Jumlah Data


                    st.write(f"Cluster Summary ({n_clusters_input} clusters):")
                    st.dataframe(cluster_summary)

                    fig, ax = plt.subplots(figsize=(12, 6))
                    scatter = ax.scatter(df['Pendidikan_RataRata'], df['Tingkat_Pengangguran'],
                                         c=df['cluster'],
                                         cmap='viridis',
                                         alpha=0.6)
                    ax.set_xlabel('Rata-rata Lama Sekolah (tahun)')
                    ax.set_ylabel('Tingkat Pengangguran (%)')
                    ax.set_title(f'Hasil Clustering Provinsi Berdasarkan Pendidikan dan Pengangguran ({n_clusters_input} Clusters)')
                    fig.colorbar(scatter, label='Cluster')
                    ax.grid(True)
                    st.pyplot(fig)
                    plt.close(fig) # Close figure

             except Exception as e:
                st.error(f"Error performing K-Means Clustering: {e}")
        else:
            st.info("Click the button to perform K-Means Clustering with the selected number of clusters.")


    # --- Linear Regression ---
    if 'Linear Regression' in models:
        st.write("#### Regresi Linear")
        try:
            # Use the full data for visualization as in your notebook
            X_linreg = df[['Pendidikan_RataRata']]
            y_linreg = df['Tingkat_Pengangguran']

            # Predict using the loaded model
            y_linreg_pred = models['Linear Regression'].predict(X_linreg)

            st.write("Regresi Linear Model Applied.")

            # Plot
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.scatter(X_linreg, y_linreg, color='blue', alpha=0.6, label='Data Asli')
            ax.plot(X_linreg, y_linreg_pred, color='red', linewidth=2, label='Regresi Linear')
            ax.set_xlabel('Rata-rata Lama Sekolah (tahun)')
            ax.set_ylabel('Tingkat Pengangguran (%)')
            ax.set_title('Regresi Linear antara Pendidikan dan Pengangguran')
            ax.legend()
            ax.grid(True)
            st.pyplot(fig)
            plt.close(fig) # Close figure

            # Calculate and display R2 and RMSE
            linreg_r2 = r2_score(y_linreg, y_linreg_pred)
            linreg_mse = mean_squared_error(y_linreg, y_linreg_pred)
            linreg_rmse = np.sqrt(linreg_mse)

            st.write("##### Model Performance")
            st.write(f"R2 Score: {linreg_r2:.4f}")
            st.write(f"RMSE: {linreg_rmse:.2f}")


        except Exception as e:
            st.error(f"Error applying Linear Regression: {e}")

    # --- Decision Tree Regression ---
    if 'Decision Tree' in models:
        st.write("#### Decision Tree Regression")
        try:
            # Use the full data for visualization/application
            X_tree = df[['Pendidikan_RataRata']]
            y_tree = df['Tingkat_Pengangguran']

            # Predict using the loaded model
            y_tree_pred = models['Decision Tree'].predict(X_tree)

            st.write("Decision Tree Model Applied.")

            # --- Visualisasi Pohon Keputusan ---
            st.subheader("Visualisasi Pohon Keputusan")
            try:
                # Decision Tree plot can be large, handle figure creation
                fig, ax = plt.subplots(figsize=(20, 10))
                # Make sure 'Pendidikan_RataRata' is the correct feature name
                plot_tree(models['Decision Tree'], feature_names=['Pendidikan_RataRata'],
                         filled=True, rounded=True, fontsize=10, ax=ax)
                ax.set_title('Visualisasi Pohon Keputusan') # Set title using ax
                st.pyplot(fig)
                plt.close(fig) # Close figure to free memory
            except Exception as e:
                st.warning(f"Could not generate Decision Tree visualization: {e}")

            # --- Text representation of the tree ---
            st.write("##### Decision Tree Structure (Text)")
            try:
                r = export_text(models['Decision Tree'], feature_names=['Pendidikan_RataRata'])
                st.text(r)
            except Exception as e:
                st.warning(f"Could not generate text tree representation: {e}")


            # --- Plot Prediksi vs Aktual ---
            st.subheader("Prediksi vs Aktual (Decision Tree)")
            # Use the 'Pendidikan_RataRata' and 'Tingkat_Pengangguran' columns from the data
            X_data = df[['Pendidikan_RataRata']]
            y_actual = df['Tingkat_Pengangguran'] # Use the actual target variable
            y_pred_dt = models['Decision Tree'].predict(X_data) # Predict using the loaded model for DT

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.scatter(X_data, y_actual, color='blue', alpha=0.5, label='Aktual')
            ax.scatter(X_data, y_pred_dt, color='green', alpha=0.5, label='Prediksi (Decision Tree)')
            ax.legend()
            ax.set_title('Perbandingan Tingkat Pengangguran Aktual vs Prediksi (Decision Tree)')
            ax.set_xlabel('Rata-rata Lama Sekolah (tahun)')
            ax.set_ylabel('Tingkat Pengangguran (%)')

             # Calculate R² and RMSE (vs actual Tingkat_Pengangguran)
            try:
                dt_r2 = r2_score(y_actual, y_pred_dt)
                dt_mse = mean_squared_error(y_actual, y_pred_dt)
                dt_rmse = np.sqrt(dt_mse)

                st.write("##### Model Performance")
                st.write(f"R2 Score: {dt_r2:.4f}")
                st.write(f"RMSE: {dt_rmse:.2f}")
            except Exception as e:
                st.warning(f"Could not calculate R² or RMSE: {e}")

            st.pyplot(fig)
            plt.close(fig) # Close figure


        except Exception as e:
            st.error(f"Error applying Decision Tree Regression: {e}")

    # --- Perbandingan Performa Model Regresi ---
    st.subheader("Perbandingan Performa Model Regresi")

    # Tampilkan metrik performa
    st.markdown(f"""
    **R² Score Linear Regression**: `{linreg_r2:.4f}`  
    **RMSE Linear Regression**: `{linreg_rmse:.2f}`

    **R² Score Decision Tree**: `{dt_r2:.4f}`  
    **RMSE Decision Tree**: `{dt_rmse:.2f}`
    """)

    # Analisis performa
    st.markdown("#### Analisis Performa:")
    if dt_r2 > linreg_r2:
        st.write("- Model Decision Tree memiliki akurasi (R²) yang lebih tinggi.")
    else:
        st.write("- Model Linear Regression memiliki akurasi (R²) yang lebih tinggi.")

    if dt_rmse < linreg_rmse:
        st.write("- Model Decision Tree menghasilkan error (RMSE) yang lebih kecil.")
    else:
        st.write("- Model Linear Regression menghasilkan error (RMSE) yang lebih kecil.")

    # Visualisasi Perbandingan R² dan RMSE
    st.markdown("#### Visualisasi Perbandingan R² dan RMSE")
    import matplotlib.pyplot as plt

    models = ['Linear Regression', 'Decision Tree']
    r2_scores = [linreg_r2, dt_r2]
    rmse_scores = [linreg_rmse, dt_rmse]

    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    # Bar plot R²
    axs[0].bar(models, r2_scores, color=['#4a90e2', '#f5a623'])
    axs[0].set_title('R² Score')
    axs[0].set_ylabel('R²')
    axs[0].set_ylim(0, 1)
    axs[0].grid(True)

    # Bar plot RMSE
    axs[1].bar(models, rmse_scores, color=['#4a90e2', '#f5a623'])
    axs[1].set_title('RMSE')
    axs[1].set_ylabel('RMSE')
    axs[1].grid(True)

    plt.suptitle('Perbandingan Performa Model')
    st.pyplot(fig)
    plt.close(fig)

    # --- Prediction Section with Button ---
    st.subheader("Make a Prediction")
    st.write("Enter a value for 'Rata-rata Lama Sekolah' to get predictions.")

    # Ensure the input value is within a reasonable range based on your data
    if df is not None and 'Pendidikan_RataRata' in df.columns:
        min_pendidikan = float(df['Pendidikan_RataRata'].min())
        max_pendidikan = float(df['Pendidikan_RataRata'].max())
        mean_pendidikan = float(df['Pendidikan_RataRata'].mean())
    else:
         # Fallback values if data is not loaded or column is missing
         min_pendidikan = 0.0
         max_pendidikan = 20.0
         mean_pendidikan = 10.0


    pendidikan_input_predict = st.slider(
        "Input Rata-rata Lama Sekolah (tahun) for Prediction:",
        min_value=min_pendidikan,
        max_value=max_pendidikan,
        value=mean_pendidikan,
        step=0.1 # Allow fractional input
    )

    # Add a button to trigger predictions
    if st.button('Get Predictions'):
        if 'Linear Regression' in models:
            try:
                # Reshape the input for prediction (needs to be 2D)
                input_for_prediction = np.array([[pendidikan_input_predict]])
                prediction_linreg = models['Linear Regression'].predict(input_for_prediction)
                st.write(f"Predicted Tingkat Pengangguran (Regresi Linear): {prediction_linreg[0]:.2f} %")
            except Exception as e:
                st.error(f"Error predicting with Linear Regression: {e}")


        if 'Decision Tree' in models:
             try:
                # Reshape the input for prediction (needs to be 2D)
                input_for_prediction = np.array([[pendidikan_input_predict]])
                prediction_tree = models['Decision Tree'].predict(input_for_prediction)
                st.write(f"Predicted Tingkat Pengangguran (Decision Tree): {prediction_tree[0]:.2f} %")
             except Exception as e:
                st.error(f"Error predicting with Decision Tree: {e}")

    else:
        st.info("Click the button to get predictions.")


# --- Conclusion ---
st.subheader("Kesimpulan dari Analisis Notebook")
st.markdown("""
Berdasarkan analisis yang dilakukan di notebook:
- *K-Means Clustering* membantu mengelompokkan provinsi berdasarkan pola pendidikan dan pengangguran.
- Model *Decision Tree* menunjukkan struktur pengambilan keputusan yang potensial untuk prediksi.
- *Regresi Linear* menunjukkan hubungan linier negatif antara rata-rata pendidikan dan tingkat pengangguran.
- Model ini dapat membantu perumusan kebijakan pendidikan dalam upaya mengurangi pengangguran di Indonesia.
""")