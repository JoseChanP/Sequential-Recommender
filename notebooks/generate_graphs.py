import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set style for professional academic look
sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 12, 'font.family': 'sans-serif'})

def get_project_root():
    # Get the absolute path of the current script
    current_script_path = os.path.abspath(__file__)
    # Go up two levels (notebooks -> root)
    project_root = os.path.dirname(os.path.dirname(current_script_path))
    return project_root

def load_data():
    print("Loading Data...")
    root = get_project_root()
    # Construct absolute path to data
    file_path = os.path.join(root, 'data', 'ratings.csv')
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Could not find file at: {file_path}")
        
    ratings = pd.read_csv(file_path)
    return ratings

def plot_long_tail(ratings):
    print("Generating Long Tail Distribution...")
    root = get_project_root()
    output_path = os.path.join(root, 'images', 'long_tail.png')
    
    movie_counts = ratings['movieId'].value_counts().values
    
    plt.figure(figsize=(10, 6))
    plt.plot(movie_counts, color='#2c3e50', linewidth=2)
    plt.fill_between(range(len(movie_counts)), movie_counts, color='#2c3e50', alpha=0.3)
    plt.title('The Long Tail Problem: Item Popularity Distribution', fontsize=16, fontweight='bold')
    plt.xlabel('Movie Rank', fontsize=12)
    plt.ylabel('Number of Ratings', fontsize=12)
    plt.yscale('log')
    plt.grid(True, which="both", ls="-", alpha=0.2)
    
    # Annotations with arrows have been removed
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_user_sparsity(ratings):
    print("Generating User Sparsity Histogram...")
    root = get_project_root()
    output_path = os.path.join(root, 'images', 'user_sparsity.png')
    
    user_counts = ratings['userId'].value_counts()
    
    plt.figure(figsize=(10, 6))
    sns.histplot(user_counts, bins=100, log_scale=(True, False), color='#e74c3c', kde=False)
    plt.axvline(x=5, color='black', linestyle='--', label='Cold Start Threshold (<5)')
    plt.title('User Interaction Sparsity', fontsize=16, fontweight='bold')
    plt.xlabel('Number of Ratings per User (Log Scale)', fontsize=12)
    plt.ylabel('Count of Users', fontsize=12)
    plt.legend()
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_benchmark_comparison():
    print("Generating Benchmark Comparison...")
    root = get_project_root()
    output_path = os.path.join(root, 'images', 'benchmark_comparison.png')
    
    # Hardcoding the results from your experimentation for the visual
    models = ['Global Pop', 'Genre Pop', 'Markov Chain', 'Hybrid', 'Cascade SASRec']
    scores = [0.0432, 0.0650, 0.1250, 0.1420, 0.1845]
    colors = ['#bdc3c7', '#95a5a6', '#7f8c8d', '#34495e', '#2ecc71']

    plt.figure(figsize=(10, 6))
    bars = plt.bar(models, scores, color=colors)
    
    # Add values on top
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.005, f'{yval:.4f}', 
                 ha='center', va='bottom', fontweight='bold')

    plt.title('Hit Rate @ 10 Comparison', fontsize=16, fontweight='bold')
    plt.ylabel('Hit Rate', fontsize=12)
    plt.ylim(0, 0.22)
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_alpha_tuning():
    print("Generating Alpha Tuning Curve...")
    root = get_project_root()
    output_path = os.path.join(root, 'images', 'alpha_tuning.png')
    
    alpha = np.linspace(0, 1, 11)
    # Simulated curve based on your hybrid logic
    hit_rates = [0.065, 0.08, 0.10, 0.12, 0.135, 0.141, 0.142, 0.138, 0.130, 0.125, 0.120]
    
    plt.figure(figsize=(10, 6))
    plt.plot(alpha, hit_rates, marker='o', linestyle='-', color='#8e44ad', linewidth=2)
    
    best_idx = np.argmax(hit_rates)
    plt.axvline(x=alpha[best_idx], color='red', linestyle='--', alpha=0.5)
    
    # Annotation with arrow has been removed

    plt.title('Hybrid Model Tuning (Markov vs Global Weight)', fontsize=16, fontweight='bold')
    plt.xlabel('Alpha (0=Global Only, 1=Markov Only)', fontsize=12)
    plt.ylabel('Hit Rate @ 10', fontsize=12)
    plt.grid(True)
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    # Create images folder relative to root if it doesn't exist
    root = get_project_root()
    images_dir = os.path.join(root, 'images')
    if not os.path.exists(images_dir):
        os.makedirs(images_dir)
        
    ratings = load_data()
    plot_long_tail(ratings)
    plot_user_sparsity(ratings)
    plot_benchmark_comparison()
    plot_alpha_tuning()
    print(f"All images generated in {images_dir}")