import pandas as pd
import matplotlib.pyplot as plt
import gzip
import os

def load_summary(path, label):
    df = pd.read_csv(path)
    df = df[df['Filter'] == 'PASS']
    df['Label'] = label
    return df

def plot_summary(standard_path, hooked_path, output_path):
    df_std = load_summary(standard_path, 'Standard')
    df_hook = load_summary(hooked_path, 'Hooked')
    df = pd.concat([df_std, df_hook])
    
    metrics = ['METRIC.Recall', 'METRIC.Precision', 'METRIC.F1_Score']
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    
    for i, metric in enumerate(metrics):
        types = df['Type'].unique()
        x = range(len(types))
        width = 0.35
        
        std_vals = [df[(df['Type'] == t) & (df['Label'] == 'Standard')][metric].values[0] for t in types]
        hook_vals = [df[(df['Type'] == t) & (df['Label'] == 'Hooked')][metric].values[0] for t in types]
        
        axes[i].bar([p - width/2 for p in x], std_vals, width, label='Standard')
        axes[i].bar([p + width/2 for p in x], hook_vals, width, label='Hooked')
        
        axes[i].set_title(metric.split('.')[-1])
        axes[i].set_xticks(x)
        axes[i].set_xticklabels(types)
        axes[i].set_ylim(0, 1.05)
        axes[i].set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        axes[i].grid(axis='y', linestyle='--', alpha=0.7)
        if i == 0:
            axes[i].legend()
    
    plt.suptitle('Hap.py Summary Metrics (Filter=PASS)')
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Saved summary plot to {output_path}")

def load_roc(path, label):
    # gzip.open is handled by pd.read_csv if compression='infer'
    df = pd.read_csv(path)
    df['Label'] = label
    return df

def plot_pr_curves(std_roc_snp, hook_roc_snp, std_roc_indel, hook_roc_indel, output_path):
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # Load all data to find global QUAL range for consistent coloring
    df_snp_std = load_roc(std_roc_snp, 'Standard')
    df_snp_hook = load_roc(hook_roc_snp, 'Hooked')
    df_indel_std = load_roc(std_roc_indel, 'Standard')
    df_indel_hook = load_roc(hook_roc_indel, 'Hooked')
    
    all_qq = pd.concat([df_snp_std['QQ'], df_snp_hook['QQ'], df_indel_std['QQ'], df_indel_hook['QQ']])
    vmin, vmax = all_qq.min(), all_qq.max()
    cmap = 'viridis'

    # SNP Subplot
    # Plot lines first so they are underneath
    axes[0].plot(df_snp_std['METRIC.Recall'], df_snp_std['METRIC.Precision'], color='blue', alpha=0.3, label='_nolegend_')
    axes[0].plot(df_snp_hook['METRIC.Recall'], df_snp_hook['METRIC.Precision'], color='red', alpha=0.3, linestyle='--', label='_nolegend_')
    
    # Standard SNP points
    sc1 = axes[0].scatter(df_snp_std['METRIC.Recall'], df_snp_std['METRIC.Precision'], 
                         c=df_snp_std['QQ'], cmap=cmap, vmin=vmin, vmax=vmax, 
                         label='Standard (o)', marker='o', s=50, edgecolors='black', linewidths=0.5)
    
    # Hooked SNP points
    axes[0].scatter(df_snp_hook['METRIC.Recall'], df_snp_hook['METRIC.Precision'], 
                   c=df_snp_hook['QQ'], cmap=cmap, vmin=vmin, vmax=vmax,
                   label='Hooked (x)', marker='x', s=60)
    
    axes[0].set_title('SNP Precision-Recall (PASS)')
    axes[0].set_xlabel('Recall')
    axes[0].set_ylabel('Precision')
    axes[0].legend()
    axes[0].grid(True, linestyle=':', alpha=0.6)
    
    # INDEL Subplot
    axes[1].plot(df_indel_std['METRIC.Recall'], df_indel_std['METRIC.Precision'], color='blue', alpha=0.3, label='_nolegend_')
    axes[1].plot(df_indel_hook['METRIC.Recall'], df_indel_hook['METRIC.Precision'], color='red', alpha=0.3, linestyle='--', label='_nolegend_')
    
    # Standard INDEL points
    axes[1].scatter(df_indel_std['METRIC.Recall'], df_indel_std['METRIC.Precision'], 
                   c=df_indel_std['QQ'], cmap=cmap, vmin=vmin, vmax=vmax, 
                   label='Standard (o)', marker='o', s=50, edgecolors='black', linewidths=0.5)
    
    # Hooked INDEL points
    axes[1].scatter(df_indel_hook['METRIC.Recall'], df_indel_hook['METRIC.Precision'], 
                   c=df_indel_hook['QQ'], cmap=cmap, vmin=vmin, vmax=vmax,
                   label='Hooked (x)', marker='x', s=60)
    
    axes[1].set_title('INDEL Precision-Recall (PASS)')
    axes[1].set_xlabel('Recall')
    axes[1].set_ylabel('Precision')
    axes[1].legend()
    axes[1].grid(True, linestyle=':', alpha=0.6)
    
    # Add a single colorbar for the whole figure
    fig.subplots_adjust(right=0.88)
    cbar_ax = fig.add_axes([0.91, 0.15, 0.02, 0.7])
    fig.colorbar(sc1, cax=cbar_ax, label='QUAL Threshold')
    
    plt.savefig(output_path)
    print(f"Saved corrected PR curves plot to {output_path}")

if __name__ == "__main__":
    base_dir = 'truth_set_benchmark'
    
    # Summary Plots
    plot_summary(
        os.path.join(base_dir, 'happy_standard/happy.output.summary.csv'),
        os.path.join(base_dir, 'happy_hooked/happy.output.summary.csv'),
        os.path.join(base_dir, 'summary_metrics.png')
    )
    
    # PR Curves
    plot_pr_curves(
        os.path.join(base_dir, 'happy_standard/happy.output.roc.Locations.SNP.PASS.csv.gz'),
        os.path.join(base_dir, 'happy_hooked/happy.output.roc.Locations.SNP.PASS.csv.gz'),
        os.path.join(base_dir, 'happy_standard/happy.output.roc.Locations.INDEL.PASS.csv.gz'),
        os.path.join(base_dir, 'happy_hooked/happy.output.roc.Locations.INDEL.PASS.csv.gz'),
        os.path.join(base_dir, 'pr_curves.png')
    )
