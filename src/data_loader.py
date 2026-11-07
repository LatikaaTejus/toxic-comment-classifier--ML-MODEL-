import os
import pandas as pd

def load_and_explore_data(csv_path: str = "data/train.csv"):
    """Loads the Jigsaw dataset and prints core distribution metrics."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found at {csv_path}. Please download it via Kaggle.")
    
    df = pd.read_csv(csv_path)
    print(f"Dataset Shape: {df.shape}")
    
    # Class breakdown
    proportions = df['toxic'].value_counts(normalize=True)
    print("Toxic Proportions:\n", proportions)
    
    # Label sums
    label_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    print("\nLabel Distribution Sums:\n", df[label_cols].sum())
    
    return df

def create_balanced_sample(df: pd.DataFrame, n_per_class: int = 50, random_state: int = 42):
    """Creates a balanced subset for zero-shot LLM evaluation."""
    toxic_sample = df[df['toxic'] == 1].sample(n=n_per_class, random_state=random_state)
    nontoxic_sample = df[df['toxic'] == 0].sample(n=n_per_class, random_state=random_state)
    
    sample_df = pd.concat([toxic_sample, nontoxic_sample]).sample(frac=1, random_state=random_state).reset_index(drop=True)
    return sample_df
