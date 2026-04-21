import streamlit as st
import torch
import pandas as pd
from datetime import datetime
from config.config import Config
from src.training.trainer import train_one_seed
from src.evaluation.evaluator import run_evaluation, get_ablation_results
from src.visualization.plot_builder import (
    build_noise_histogram, build_prediction_plot, 
    build_ablation_plot, build_comparison_grid
)

def main():
    st.title("Conditional LSTM Bandpass Filter")
    
    # Sidebar Configuration
    st.sidebar.header("Hyperparameters")
    freqs = st.sidebar.multiselect("Frequencies (Hz)", [1, 3, 5, 7, 9], default=[1, 3, 5, 7])
    window = st.sidebar.slider("Context Window", 10, 200, 100)
    epochs = st.sidebar.slider("Epochs", 1, 10, 3)
    hidden_size = st.sidebar.number_input("Hidden Size", 16, 256, 64)
    lr = st.sidebar.number_input("Learning Rate", 0.0001, 0.01, 0.001, format="%.4f")
    
    config = Config(
        FREQUENCIES=freqs,
        CONTEXT_WINDOW=window,
        EPOCHS=epochs,
        HIDDEN_SIZE=hidden_size,
        LEARNING_RATE=lr
    )
    
    if st.sidebar.button("Run Experiment"):
        st.info("Training models and generating results...")
        
        # 1. Noise Histogram
        st.subheader("1. Noise Distribution Independence")
        fig_noise = build_noise_histogram(config)
        st.pyplot(fig_noise)
        
        # 2. Run Main Experiment (L=1 vs L=100)
        results = []
        
        # L=1 Model
        config.L_PARAMETER = 1
        model_l1 = train_one_seed(config, seed=0)
        metrics_l1 = run_evaluation(model_l1, config)
        results.append({"Variant": "L=1", "MSE": metrics_l1['mse']})
        
        # L=100 Model
        config.L_PARAMETER = 100
        model_l100 = train_one_seed(config, seed=0)
        metrics_l100 = run_evaluation(model_l100, config)
        results.append({"Variant": "L=100", "MSE": metrics_l100['mse']})
        
        # Display Metrics
        st.subheader("2. Quantitative Metrics")
        df_results = pd.DataFrame(results)
        st.dataframe(df_results)
        
        # 3. Prediction Plots
        st.subheader("3. Frequency Extraction (L=1)")
        for i in range(len(config.FREQUENCIES)):
            fig_pred = build_prediction_plot(model_l1, config, i)
            st.pyplot(fig_pred)
            
        # 4. Comparison Grid
        st.subheader("4. L=1 vs L=100 Comparison Grid")
        fig_grid = build_comparison_grid(model_l1, model_l100, config)
        st.pyplot(fig_grid)
        
        # 5. Targeted Ablation
        st.subheader("5. Targeted Saliency Ablation (1Hz vs 7Hz)")
        abl_results = get_ablation_results(model_l1, config)
        fig_abl = build_ablation_plot(abl_results)
        st.pyplot(fig_abl)
        
        # Auto-save primary figure
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        save_path = config.SCREENSHOTS_DIR / f"results_{timestamp}.png"
        fig_grid.savefig(save_path)
        st.success(f"Primary comparison grid saved to {save_path}")

if __name__ == "__main__":
    main()
