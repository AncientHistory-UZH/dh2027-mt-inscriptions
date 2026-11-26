import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

def generate_visualizations(metrics_csv="data/evaluation_metrics.csv", output_dir="visualizations"):
    if not os.path.exists(metrics_csv):
        print(f"Metrics file {metrics_csv} not found. Ensure evaluation has run.")
        return
        
    df = pd.read_csv(metrics_csv)
    os.makedirs(output_dir, exist_ok=True)
    
    # Set scientific standard styling
    sns.set_theme(style="whitegrid", context="paper")
    plt.rc('font', family='serif')
    
    metrics = ["BLEU", "chrF", "BERTScore_F1", "BLEURT", "COMET"]
    
    for metric in metrics:
        if metric not in df.columns:
            continue
            
        plt.figure(figsize=(10, 6))
        # Create grouped barplot
        ax = sns.barplot(
            data=df, 
            x="Language", 
            y=metric, 
            hue="Model",
            palette="viridis",
            errorbar=None
        )
        
        plt.title(f"{metric} Scores by Model and Target Language", fontsize=14, pad=15)
        plt.xlabel("Target Language", fontsize=12)
        plt.ylabel(metric, fontsize=12)
        
        # Place legend outside
        plt.legend(title="Model", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        # Save high-res PNG and PDF for publishing
        plt.savefig(os.path.join(output_dir, f"{metric}_comparison.pdf"), dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(output_dir, f"{metric}_comparison.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
    print(f"High-quality visualizations saved to {output_dir}/")

if __name__ == "__main__":
    generate_visualizations()
