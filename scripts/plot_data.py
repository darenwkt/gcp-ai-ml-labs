import pandas as pd
import matplotlib.pyplot as plt
import os

def main():
    # Load the datasets
    train_df = pd.read_csv("data/training_data.csv")
    serve_df = pd.read_csv("data/serving_data_skewed.csv")

    # Set up dark theme visualization
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(10, 8), dpi=150)

    # Plot normal training data
    ax.scatter(
        train_df['feature1'], train_df['feature2'],
        color='#f43f5e', alpha=0.5, label='Training Data (Normal Distribution)',
        edgecolors='none', s=25
    )

    # Plot skewed serving data
    ax.scatter(
        serve_df['feature1'], serve_df['feature2'],
        color='#3b82f6', alpha=0.6, label='Serving Data (Skewed/Drifted)',
        edgecolors='none', s=25
    )

    # Customize grid & lines
    ax.grid(True, linestyle='--', alpha=0.2, color='#9ca3af')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#4b5563')
    ax.spines['bottom'].set_color('#4b5563')

    # Add labels, legend and title
    ax.set_title('Training vs. Serving Data Distribution', fontsize=16, fontweight='bold', pad=20, color='#f3f4f6')
    ax.set_xlabel('Feature 1', fontsize=12, labelpad=10, color='#d1d5db')
    ax.set_ylabel('Feature 2', fontsize=12, labelpad=10, color='#d1d5db')
    ax.tick_params(colors='#9ca3af')

    legend = ax.legend(frameon=True, facecolor='#1f2937', edgecolor='#4b5563', fontsize=11, loc='upper right')
    for text in legend.get_texts():
        text.set_color('#f3f4f6')

    plt.tight_layout()

    # Save to the artifacts directory
    artifact_dir = "/Users/darenwkt/.gemini/jetski/brain/84ea1d05-b3cc-4108-a3f5-28e23c531a98"
    os.makedirs(artifact_dir, exist_ok=True)
    save_path = os.path.join(artifact_dir, "dataset_comparison.png")
    
    plt.savefig(save_path, bbox_inches='tight')
    print(f"Plot saved successfully to {save_path}")

if __name__ == "__main__":
    main()
