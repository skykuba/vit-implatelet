import os
import sys
import subprocess
import pandas as pd
import numpy as np

# Import Python normalization
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from normalize_deseq import normalize_deseq2_no_report, load_data

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_file = os.path.join(script_dir, '../../../data/raw/counts_raw.csv')
    
    print("Loading input data...")
    counts_raw = load_data(input_file)
    counts_raw = counts_raw.apply(pd.to_numeric, errors='coerce').dropna()
    
    # Subsetting to 5000 genes to prevent Out Of Memory (OOM) issues during testing
    # Needs to be >1000 genes with count>5 for R's vst nsub=1000
    counts_raw = counts_raw.head(5000)
    
    # Save subsetted data to calculate R normalization
    temp_input = os.path.join(script_dir, 'temp_subset_counts.csv')
    counts_raw.to_csv(temp_input)

    print("\n--- Running Python normalization (fast=True) ---")
    python_norm = normalize_deseq2_no_report(counts_raw, fast=True, n_cpus=2)

    print("\n--- Running R normalization (fast=TRUE) ---")
    r_wrapper = os.path.join(script_dir, 'run_r_fast.R')
    r_output = os.path.join(script_dir, '../../../data/output/normalized_counts_R_fast.tsv')
    
    # Create R wrapper file
    with open(r_wrapper, 'w') as f:
        f.write(f'''
source("{os.path.join(script_dir, 'statisticalAnalysis.R')}")
data <- read.csv("{temp_input}", row.names=1)
# R normalizeDESeq2NoReport takes matrix where rows are genes and cols are samples
normalized <- normalizeDESeq2NoReport(as.matrix(data), fast=TRUE)
write.table(normalized, file="{r_output}", sep="\\t", quote=FALSE, col.names=NA)
''')

    # Try to execute R script
    try:
        subprocess.run(['Rscript', r_wrapper], check=True)
        print("R normalization completed successfully.")
    except Exception as e:
        print(f"Error executing R script: {e}")
        return

    # Delete wrapper (cleanup)
    os.remove(r_wrapper)
    os.remove(temp_input)

    print("\n--- Comparison ---")
    r_norm = pd.read_csv(r_output, sep='\t', index_col=0)
    
    # Python has genes in index, samples in columns. R returns the same.
    python_norm, r_norm = python_norm.align(r_norm)
    
    # Check if column names were prefixed by R (e.g. starting with X, or having '.' instead of '-')
    if len(python_norm.columns) > 0 and len(r_norm.columns) > 0:
        # Pydeseq2 return and R return index/cols might have string mismatch
        # Fix column names if R replaced dashes with dots
        r_cols_fixed = [c.replace('.', '-') if isinstance(c, str) else c for c in r_norm.columns]
        if 'X' in [str(c)[0] for c in r_norm.columns]:
            r_cols_fixed = [c[1:] if str(c).startswith('X') else c for c in r_cols_fixed]
        
        # Compare columns and indexes directly without aligning to avoid introducing naive NaNs
        r_norm_loaded = pd.read_csv(r_output, sep='\t', index_col=0)
        r_norm_loaded.columns = r_norm_loaded.columns.str.replace('.', '-').str.replace('X', '', regex=False)
        
        # Recreate the dataframe based on common genes and samples
        common_genes = python_norm.index.intersection(r_norm_loaded.index)
        common_samples = python_norm.columns.intersection(r_norm_loaded.columns)
        
        py_subset = python_norm.loc[common_genes, common_samples]
        r_subset = r_norm_loaded.loc[common_genes, common_samples]
        
        diff = (py_subset - r_subset).abs()
        max_diff = diff.max().max()
        mean_diff = diff.mean().mean()
        
        print(f"Number of compared genes: {len(common_genes)} | samples: {len(common_samples)}")
        
        # Add correlation calculation
        py_flat = py_subset.values.flatten()
        r_flat = r_subset.values.flatten()
        
        # Pearson
        pearson_corr = np.corrcoef(py_flat, r_flat)[0, 1]
        
        # Spearman
        from scipy.stats import spearmanr
        spearman_corr, _ = spearmanr(py_flat, r_flat)
        
        print(f"Pearson Correlation: {pearson_corr:.5f}")
        print(f"Spearman Correlation: {spearman_corr:.5f}")
        
        print("\n--- Sample of 10 random values for comparison ---")
        # Select 10 random pairs (gene, sample)
        np.random.seed(42) # For reproducibility
        random_genes = np.random.choice(common_genes, 10, replace=False)
        random_samples = np.random.choice(common_samples, 10, replace=False)
        
        comparison_df = pd.DataFrame(index=range(10), columns=['Gene', 'Sample', 'Python', 'R', 'Difference'])
        for i, (g, s) in enumerate(zip(random_genes, random_samples)):
            val_py = py_subset.loc[g, s]
            val_r = r_subset.loc[g, s]
            comparison_df.loc[i] = [g, s, f"{val_py:.4f}", f"{val_r:.4f}", f"{abs(val_py - val_r):.4f}"]
        
        print(comparison_df.to_string(index=False))
        print("--------------------------------------------------\n")
    
    print(f"Max absolute difference (Max diff): {max_diff}")
    print(f"Mean absolute difference (Mean diff): {mean_diff}")
    
    if pearson_corr >= 0.95:
        print("SUCCESS: Normalization results are VERY SIMILAR (Correlation p>=0.95).")
    elif max_diff < 1e-5:
        print("Normalization results are IDENTICAL (or very similar).")
    else:
        print("Normalization results are DIFFERENT.")

if __name__ == '__main__':
    main()
