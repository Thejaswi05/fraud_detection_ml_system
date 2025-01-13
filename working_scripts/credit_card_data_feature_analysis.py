import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

def analyze_important_features(df, feature_list=['V15', 'V11', 'V13', 'V18', 'V5']):
    # Create figure with subplots
    fig = plt.figure(figsize=(15, 10))
    
    # 1. Distribution plot for top features
    for idx, feature in enumerate(feature_list, 1):
        plt.subplot(2, 3, idx)
        # Plot distribution for normal vs fraud transactions
        sns.kdeplot(data=df[df['Class'] == 0][feature], label='Normal', color='blue', alpha=0.5)
        sns.kdeplot(data=df[df['Class'] == 1][feature], label='Fraud', color='red', alpha=0.5)
        plt.title(f'{feature} Distribution')
        plt.legend()
    
    # 2. Correlation heatmap
    plt.subplot(2, 3, 6)
    correlation = df[feature_list].corr()
    sns.heatmap(correlation, annot=True, cmap='coolwarm', center=0)
    plt.title('Feature Correlations')
    
    plt.tight_layout()
    return plt

def calculate_feature_stats(df, feature_list=['V15', 'V11', 'V13', 'V18', 'V5']):
    stats_dict = {}
    
    for feature in feature_list:
        normal_trans = df[df['Class'] == 0][feature]
        fraud_trans = df[df['Class'] == 1][feature]
        
        # Calculate statistics
        stats_dict[feature] = {
            'normal_mean': normal_trans.mean(),
            'fraud_mean': fraud_trans.mean(),
            'normal_std': normal_trans.std(),
            'fraud_std': fraud_trans.std(),
            'ks_statistic': stats.ks_2samp(normal_trans, fraud_trans).statistic
        }
    
    return pd.DataFrame(stats_dict).round(3)

# Load your data and run analysis
df = pd.read_csv('creditcard.csv')
plt = analyze_important_features(df)
plt.savefig('feature_analysis.png')
print("\nFeature Statistics:")
print(calculate_feature_stats(df))