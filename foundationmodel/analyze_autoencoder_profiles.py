#!/usr/bin/env python3
"""
Analyze Autoencoder Layer Profiles

Loads profile metrics from foundationmodel/profile_with_autoencoders.py output and generates:
- Histograms of PCA explained variances
- Histograms of autoencoder MSE performance
- Scatterplots: layer size vs autoencoder performance
- Breakdowns by layer type (conv, dense, transformer, norm, other)

Usage:
    python analyze_autoencoder_profiles.py <profile_dir>
    python analyze_autoencoder_profiles.py ./autoencoder_profiles_20260203_000534
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 5)


# ============================================================================
# Layer Type Classification
# ============================================================================

def classify_layer_type(layer_name: str) -> str:
    """
    Classify layer into type: conv, dense, transformer, norm, or other.
    """
    name_lower = layer_name.lower()

    # Transformer layers
    if any(x in name_lower for x in ['attention', 'encoder_layer', 'intermediate',
                                       'position_feed_forward', 'feat_extractor']):
        return 'transformer'

    # Convolutional layers
    if any(x in name_lower for x in ['conv', 'convolution', 'patch_extractor',
                                       'pooling', 'depthwise']):
        return 'conv'

    # Dense/Linear layers
    if any(x in name_lower for x in ['linear', 'dense', 'gemm', 'output_dense',
                                       'attention_output_dense']):
        return 'dense'

    # Normalization layers
    if any(x in name_lower for x in ['layernorm', 'layer_norm', 'batchnorm', 'batch_norm']):
        return 'norm'

    # Pooler
    if 'pooler' in name_lower:
        return 'pooling'

    return 'other'


# ============================================================================
# Data Loading
# ============================================================================

def load_profile_metrics(profile_dir: Path) -> Dict:
    """
    Load all metrics.json files from a profile directory.
    Extracts original layer dimensions from PCA objects in .pkl files.

    Returns:
        Dict with structure:
        {
            'model_name': {
                'layers': [
                    {
                        'layer_name': str,
                        'layer_type': str (module class),
                        'layer_category': str (conv/dense/transformer/norm/pooling/other),
                        'input_size': int (original activation features),
                        'output_size': int (original activation features),
                        'input_pca_var': float,
                        'output_pca_var': float,
                        'train_mse_normalized': float,
                        'val_mse': float,
                    },
                    ...
                ]
            }
        }
    """
    import pickle
    results = {}

    # Find all model directories
    for model_dir in sorted(profile_dir.iterdir()):
        if not model_dir.is_dir():
            continue

        model_name = model_dir.name
        print(f"Loading metrics from {model_name}...")

        layers = []
        metrics_files = sorted(model_dir.glob('*_metrics.json'))

        for metrics_file in metrics_files:
            try:
                with open(metrics_file, 'r') as f:
                    data = json.load(f)

                # Extract original layer sizes from PCA objects in .pkl file
                pkl_file = metrics_file.with_suffix('.pkl')
                input_size = None
                output_size = None

                if pkl_file.exists():
                    try:
                        with open(pkl_file, 'rb') as f:
                            profile = pickle.load(f)

                        # Get original feature dimensions from PCA objects
                        input_pca = profile.get('input_pca')
                        output_pca = profile.get('output_pca')

                        if input_pca is not None and hasattr(input_pca, 'n_features_in_'):
                            input_size = input_pca.n_features_in_
                        if output_pca is not None and hasattr(output_pca, 'n_features_in_'):
                            output_size = output_pca.n_features_in_
                    except Exception as e:
                        pass

                # Fallback to compressed shape if PCA extraction fails
                if input_size is None:
                    input_shape = data.get('input_shape_sample', [0])
                    input_size = int(np.prod(input_shape)) if input_shape else 0
                if output_size is None:
                    output_shape = data.get('output_shape_sample', [0])
                    output_size = int(np.prod(output_shape)) if output_shape else 0

                layer_info = {
                    'layer_name': data.get('layer_name', 'unknown'),
                    'layer_type': data.get('layer_type', 'Unknown'),
                    'layer_category': classify_layer_type(data.get('layer_name', '')),
                    'input_size': input_size,
                    'output_size': output_size,
                    'input_pca_var': data.get('input_pca_explained_variance', 0.0),
                    'output_pca_var': data.get('output_pca_explained_variance', 0.0),
                    'train_mse': data.get('train_mse', 0.0),
                    'train_mse_normalized': data.get('train_mse_normalized', 0.0),
                    'val_mse': data.get('val_mse', 0.0),
                    'epochs_trained': data.get('epochs_trained', 0),
                }

                layers.append(layer_info)

            except Exception as e:
                print(f"  Warning: Failed to load {metrics_file.name}: {e}")
                continue

        if layers:
            results[model_name] = {'layers': layers}
            print(f"  Loaded {len(layers)} layers from {model_name}")

    return results


# ============================================================================
# Analysis Functions
# ============================================================================

def get_metrics_by_category(layers: List[Dict], category: str = None) -> Dict[str, np.ndarray]:
    """Extract metrics grouped by category."""
    if category:
        layers = [l for l in layers if l['layer_category'] == category]

    return {
        'input_pca': np.array([l['input_pca_var'] for l in layers]),
        'output_pca': np.array([l['output_pca_var'] for l in layers]),
        'mse_normalized': np.array([l['train_mse_normalized'] for l in layers]),
        'input_size': np.array([l['input_size'] for l in layers]),
        'output_size': np.array([l['output_size'] for l in layers]),
    }


def print_statistics(layers: List[Dict], category: str = None):
    """Print summary statistics."""
    if category:
        layers = [l for l in layers if l['layer_category'] == category]
        print(f"\n{category.upper()} LAYERS ({len(layers)} total):")
    else:
        print(f"\nALL LAYERS ({len(layers)} total):")

    metrics = get_metrics_by_category(layers, None)

    print(f"  Input PCA Var:    mean={metrics['input_pca'].mean():.4f}, "
          f"std={metrics['input_pca'].std():.4f}, "
          f"min={metrics['input_pca'].min():.4f}, max={metrics['input_pca'].max():.4f}")
    print(f"  Output PCA Var:   mean={metrics['output_pca'].mean():.4f}, "
          f"std={metrics['output_pca'].std():.4f}, "
          f"min={metrics['output_pca'].min():.4f}, max={metrics['output_pca'].max():.4f}")
    print(f"  MSE (normalized): mean={metrics['mse_normalized'].mean():.6f}, "
          f"std={metrics['mse_normalized'].std():.6f}, "
          f"min={metrics['mse_normalized'].min():.6f}, max={metrics['mse_normalized'].max():.6f}")
    print(f"  Layer size (out): mean={metrics['output_size'].mean():.0f}, "
          f"median={np.median(metrics['output_size']):.0f}, "
          f"max={metrics['output_size'].max():.0f}")


# ============================================================================
# Visualization Functions
# ============================================================================

def create_histograms(layers: List[Dict], output_dir: Path):
    """Create histogram visualizations."""
    print("\nGenerating histograms...")

    metrics = get_metrics_by_category(layers)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('PCA Explained Variance & Autoencoder Performance Distribution', fontsize=14, fontweight='bold')

    # Input PCA explained variance
    axes[0, 0].hist(metrics['input_pca'], bins=20, color='steelblue', edgecolor='black', alpha=0.7)
    axes[0, 0].set_xlabel('Explained Variance Ratio')
    axes[0, 0].set_ylabel('Number of Layers')
    axes[0, 0].set_title('Input PCA Explained Variance')
    axes[0, 0].axvline(metrics['input_pca'].mean(), color='red', linestyle='--', linewidth=2, label=f"Mean: {metrics['input_pca'].mean():.3f}")
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)

    # Output PCA explained variance
    axes[0, 1].hist(metrics['output_pca'], bins=20, color='seagreen', edgecolor='black', alpha=0.7)
    axes[0, 1].set_xlabel('Explained Variance Ratio')
    axes[0, 1].set_ylabel('Number of Layers')
    axes[0, 1].set_title('Output PCA Explained Variance')
    axes[0, 1].axvline(metrics['output_pca'].mean(), color='red', linestyle='--', linewidth=2, label=f"Mean: {metrics['output_pca'].mean():.3f}")
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)

    # MSE (normalized)
    mse_nonzero = metrics['mse_normalized'][metrics['mse_normalized'] > 0]
    axes[1, 0].hist(mse_nonzero, bins=20, color='coral', edgecolor='black', alpha=0.7)
    axes[1, 0].set_xlabel('Normalized MSE')
    axes[1, 0].set_ylabel('Number of Layers')
    axes[1, 0].set_title('Autoencoder MSE (normalized by output variance)')
    axes[1, 0].axvline(mse_nonzero.mean(), color='red', linestyle='--', linewidth=2, label=f"Mean: {mse_nonzero.mean():.6f}")
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)

    # MSE log scale
    axes[1, 1].hist(mse_nonzero, bins=20, color='coral', edgecolor='black', alpha=0.7)
    axes[1, 1].set_xlabel('Normalized MSE')
    axes[1, 1].set_ylabel('Number of Layers')
    axes[1, 1].set_title('Autoencoder MSE (log scale)')
    axes[1, 1].set_yscale('log')
    axes[1, 1].grid(alpha=0.3)

    plt.tight_layout()
    histogram_file = output_dir / 'histograms_all.png'
    plt.savefig(histogram_file, dpi=150, bbox_inches='tight')
    print(f"  Saved: {histogram_file}")
    plt.close()


def create_scatterplots(layers: List[Dict], output_dir: Path):
    """Create scatterplot visualizations."""
    print("\nGenerating scatterplots...")

    metrics = get_metrics_by_category(layers)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Layer Size vs Performance Metrics', fontsize=14, fontweight='bold')

    # Filter out zero sizes for log scale
    valid_mask = metrics['output_size'] > 0
    output_size = metrics['output_size'][valid_mask]
    input_pca = metrics['input_pca'][valid_mask]
    output_pca = metrics['output_pca'][valid_mask]
    mse = metrics['mse_normalized'][valid_mask]

    # Output size vs Input PCA var
    axes[0, 0].scatter(output_size, input_pca, alpha=0.6, s=50, color='steelblue')
    axes[0, 0].set_xlabel('Layer Output Size (log scale)')
    axes[0, 0].set_ylabel('Input PCA Explained Variance')
    axes[0, 0].set_title('Output Size vs Input PCA Variance')
    axes[0, 0].set_xscale('log')
    axes[0, 0].grid(alpha=0.3)

    # Output size vs Output PCA var
    axes[0, 1].scatter(output_size, output_pca, alpha=0.6, s=50, color='seagreen')
    axes[0, 1].set_xlabel('Layer Output Size (log scale)')
    axes[0, 1].set_ylabel('Output PCA Explained Variance')
    axes[0, 1].set_title('Output Size vs Output PCA Variance')
    axes[0, 1].set_xscale('log')
    axes[0, 1].grid(alpha=0.3)

    # Output size vs MSE
    axes[1, 0].scatter(output_size, mse, alpha=0.6, s=50, color='coral')
    axes[1, 0].set_xlabel('Layer Output Size (log scale)')
    axes[1, 0].set_ylabel('Normalized MSE')
    axes[1, 0].set_title('Output Size vs Autoencoder MSE')
    axes[1, 0].set_xscale('log')
    axes[1, 0].set_yscale('log')
    axes[1, 0].grid(alpha=0.3)

    # Input size vs Output size (relationship)
    input_size = metrics['input_size'][valid_mask]
    axes[1, 1].scatter(input_size, output_size, alpha=0.6, s=50, color='purple')
    axes[1, 1].set_xlabel('Layer Input Size (log scale)')
    axes[1, 1].set_ylabel('Layer Output Size (log scale)')
    axes[1, 1].set_title('Input vs Output Layer Sizes')
    axes[1, 1].set_xscale('log')
    axes[1, 1].set_yscale('log')
    axes[1, 1].grid(alpha=0.3)

    plt.tight_layout()
    scatterplot_file = output_dir / 'scatterplots_all.png'
    plt.savefig(scatterplot_file, dpi=150, bbox_inches='tight')
    print(f"  Saved: {scatterplot_file}")
    plt.close()


def create_category_histograms(layers: List[Dict], output_dir: Path):
    """Create histograms broken down by layer type."""
    print("\nGenerating category-specific histograms...")

    categories = ['conv', 'dense', 'transformer', 'norm', 'pooling', 'other']

    # PCA variance by category
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('PCA Explained Variance by Layer Type', fontsize=14, fontweight='bold')

    input_pca_data = [get_metrics_by_category(layers, cat)['input_pca'] for cat in categories]
    output_pca_data = [get_metrics_by_category(layers, cat)['output_pca'] for cat in categories]

    axes[0].boxplot(input_pca_data, labels=categories)
    axes[0].set_ylabel('Explained Variance Ratio')
    axes[0].set_title('Input PCA Variance by Layer Type')
    axes[0].grid(alpha=0.3, axis='y')
    plt.setp(axes[0].xaxis.get_majorticklabels(), rotation=45, ha='right')

    axes[1].boxplot(output_pca_data, labels=categories)
    axes[1].set_ylabel('Explained Variance Ratio')
    axes[1].set_title('Output PCA Variance by Layer Type')
    axes[1].grid(alpha=0.3, axis='y')
    plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=45, ha='right')

    plt.tight_layout()
    category_pca_file = output_dir / 'category_pca_variance.png'
    plt.savefig(category_pca_file, dpi=150, bbox_inches='tight')
    print(f"  Saved: {category_pca_file}")
    plt.close()

    # MSE by category
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle('Autoencoder MSE by Layer Type', fontsize=14, fontweight='bold')

    mse_data = [get_metrics_by_category(layers, cat)['mse_normalized'] for cat in categories]
    ax.boxplot(mse_data, labels=categories)
    ax.set_ylabel('Normalized MSE')
    ax.set_title('Autoencoder Performance by Layer Type')
    ax.set_yscale('log')
    ax.grid(alpha=0.3, axis='y')
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    plt.tight_layout()
    category_mse_file = output_dir / 'category_mse.png'
    plt.savefig(category_mse_file, dpi=150, bbox_inches='tight')
    print(f"  Saved: {category_mse_file}")
    plt.close()


def create_category_scatterplots(layers: List[Dict], output_dir: Path):
    """Create scatterplots broken down by layer type."""
    print("\nGenerating category-specific scatterplots...")

    categories = ['conv', 'dense', 'transformer', 'norm', 'pooling', 'other']
    colors = {'conv': 'blue', 'dense': 'green', 'transformer': 'red',
              'norm': 'orange', 'pooling': 'purple', 'other': 'gray'}

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Layer Size vs Performance by Type', fontsize=14, fontweight='bold')

    # Output size vs Output PCA variance
    for category in categories:
        metrics = get_metrics_by_category(layers, category)
        valid_mask = metrics['output_size'] > 0
        output_size = metrics['output_size'][valid_mask]
        output_pca = metrics['output_pca'][valid_mask]

        if len(output_size) > 0:
            axes[0].scatter(output_size, output_pca, alpha=0.6, s=50,
                          color=colors[category], label=f"{category} (n={len(output_size)})")

    axes[0].set_xlabel('Layer Output Size (log scale)')
    axes[0].set_ylabel('Output PCA Explained Variance')
    axes[0].set_title('Output Size vs Output PCA Variance')
    axes[0].set_xscale('log')
    axes[0].legend(loc='best', fontsize=9)
    axes[0].grid(alpha=0.3)

    # Output size vs MSE
    for category in categories:
        metrics = get_metrics_by_category(layers, category)
        valid_mask = metrics['output_size'] > 0
        output_size = metrics['output_size'][valid_mask]
        mse = metrics['mse_normalized'][valid_mask]

        if len(output_size) > 0:
            axes[1].scatter(output_size, mse, alpha=0.6, s=50,
                          color=colors[category], label=f"{category} (n={len(output_size)})")

    axes[1].set_xlabel('Layer Output Size (log scale)')
    axes[1].set_ylabel('Normalized MSE')
    axes[1].set_title('Output Size vs Autoencoder MSE')
    axes[1].set_xscale('log')
    axes[1].set_yscale('log')
    axes[1].legend(loc='best', fontsize=9)
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    category_scatter_file = output_dir / 'category_scatterplots.png'
    plt.savefig(category_scatter_file, dpi=150, bbox_inches='tight')
    print(f"  Saved: {category_scatter_file}")
    plt.close()


def create_layer_type_analysis(layers: List[Dict], output_dir: Path):
    """Create analysis of layer types (PyTorch module types)."""
    print("\nGenerating layer type analysis...")

    # Group by actual layer type
    layer_types = defaultdict(list)
    for layer in layers:
        layer_types[layer['layer_type']].append(layer)

    # Create table of statistics
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('off')

    table_data = []
    table_data.append(['Layer Type', 'Count', 'Avg MSE', 'Avg Output PCA', 'Avg Layer Size'])

    for layer_type in sorted(layer_types.keys()):
        layers_of_type = layer_types[layer_type]
        mse_vals = np.array([l['train_mse_normalized'] for l in layers_of_type])
        pca_vals = np.array([l['output_pca_var'] for l in layers_of_type])
        size_vals = np.array([l['output_size'] for l in layers_of_type])

        table_data.append([
            layer_type[:30],  # Truncate long names
            str(len(layers_of_type)),
            f"{mse_vals.mean():.6f}",
            f"{pca_vals.mean():.4f}",
            f"{size_vals.mean():.0f}"
        ])

    table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                    colWidths=[0.25, 0.1, 0.2, 0.2, 0.2])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)

    # Header styling
    for i in range(len(table_data[0])):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # Alternate row colors
    for i in range(1, len(table_data)):
        color = '#f0f0f0' if i % 2 == 0 else 'white'
        for j in range(len(table_data[0])):
            table[(i, j)].set_facecolor(color)

    plt.title('Layer Type Statistics', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    layer_type_file = output_dir / 'layer_type_statistics.png'
    plt.savefig(layer_type_file, dpi=150, bbox_inches='tight')
    print(f"  Saved: {layer_type_file}")
    plt.close()


# ============================================================================
# Aggregate Analysis (across all models)
# ============================================================================

def create_aggregate_histograms(results: Dict, output_dir: Path):
    """Create histograms aggregating all models together."""
    print("\nGenerating aggregate histograms across all models...")

    # Collect all layers from all models
    all_layers = []
    for model_name, data in results.items():
        all_layers.extend(data['layers'])

    metrics = get_metrics_by_category(all_layers)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('All Models: PCA Explained Variance & Autoencoder Performance',
                 fontsize=14, fontweight='bold')

    # Input PCA explained variance
    axes[0, 0].hist(metrics['input_pca'], bins=25, color='steelblue', edgecolor='black', alpha=0.7)
    axes[0, 0].set_xlabel('Explained Variance Ratio')
    axes[0, 0].set_ylabel('Number of Layers')
    axes[0, 0].set_title(f'Input PCA Variance (n={len(metrics["input_pca"])})')
    axes[0, 0].axvline(metrics['input_pca'].mean(), color='red', linestyle='--', linewidth=2,
                       label=f"Mean: {metrics['input_pca'].mean():.3f}")
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)

    # Output PCA explained variance
    axes[0, 1].hist(metrics['output_pca'], bins=25, color='seagreen', edgecolor='black', alpha=0.7)
    axes[0, 1].set_xlabel('Explained Variance Ratio')
    axes[0, 1].set_ylabel('Number of Layers')
    axes[0, 1].set_title(f'Output PCA Variance (n={len(metrics["output_pca"])})')
    axes[0, 1].axvline(metrics['output_pca'].mean(), color='red', linestyle='--', linewidth=2,
                       label=f"Mean: {metrics['output_pca'].mean():.3f}")
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)

    # MSE (normalized)
    mse_nonzero = metrics['mse_normalized'][metrics['mse_normalized'] > 0]
    axes[1, 0].hist(mse_nonzero, bins=25, color='coral', edgecolor='black', alpha=0.7)
    axes[1, 0].set_xlabel('Normalized MSE')
    axes[1, 0].set_ylabel('Number of Layers')
    axes[1, 0].set_title(f'Autoencoder MSE (normalized, n={len(mse_nonzero)})')
    axes[1, 0].axvline(mse_nonzero.mean(), color='red', linestyle='--', linewidth=2,
                       label=f"Mean: {mse_nonzero.mean():.6f}")
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)

    # MSE log scale
    axes[1, 1].hist(mse_nonzero, bins=25, color='coral', edgecolor='black', alpha=0.7)
    axes[1, 1].set_xlabel('Normalized MSE')
    axes[1, 1].set_ylabel('Number of Layers (log scale)')
    axes[1, 1].set_title('Autoencoder MSE Distribution (log scale)')
    axes[1, 1].set_yscale('log')
    axes[1, 1].grid(alpha=0.3)

    plt.tight_layout()
    file = output_dir / '00_aggregate_histograms.png'
    plt.savefig(file, dpi=150, bbox_inches='tight')
    print(f"  Saved: {file}")
    plt.close()


def create_aggregate_scatterplots(results: Dict, output_dir: Path):
    """Create scatterplots with different colors for each model."""
    print("\nGenerating aggregate scatterplots across all models...")

    colors = {}
    color_palette = plt.cm.tab10(np.linspace(0, 1, len(results)))
    for i, model_name in enumerate(sorted(results.keys())):
        colors[model_name] = color_palette[i]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('All Models: Layer Size vs Performance', fontsize=14, fontweight='bold')

    for model_name, data in results.items():
        layers = data['layers']
        metrics = get_metrics_by_category(layers)
        valid_mask = metrics['output_size'] > 0
        output_size = metrics['output_size'][valid_mask]
        output_pca = metrics['output_pca'][valid_mask]
        mse = metrics['mse_normalized'][valid_mask]

        # Output size vs Output PCA
        axes[0].scatter(output_size, output_pca, alpha=0.6, s=60,
                       color=colors[model_name], label=f"{model_name} (n={len(output_size)})")

        # Output size vs MSE
        axes[1].scatter(output_size, mse, alpha=0.6, s=60,
                       color=colors[model_name], label=f"{model_name} (n={len(output_size)})")

    axes[0].set_xlabel('Layer Output Size (log scale)')
    axes[0].set_ylabel('Output PCA Explained Variance')
    axes[0].set_title('Size vs Output PCA Variance')
    axes[0].set_xscale('log')
    axes[0].legend(loc='best', fontsize=9)
    axes[0].grid(alpha=0.3)

    axes[1].set_xlabel('Layer Output Size (log scale)')
    axes[1].set_ylabel('Normalized MSE')
    axes[1].set_title('Size vs Autoencoder MSE')
    axes[1].set_xscale('log')
    axes[1].set_yscale('log')
    axes[1].legend(loc='best', fontsize=9)
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    file = output_dir / '00_aggregate_scatterplots.png'
    plt.savefig(file, dpi=150, bbox_inches='tight')
    print(f"  Saved: {file}")
    plt.close()


def create_aggregate_model_comparison(results: Dict, output_dir: Path):
    """Create comparison plots between models."""
    print("\nGenerating model comparison plots...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Model Comparison', fontsize=14, fontweight='bold')

    model_names = sorted(results.keys())
    model_stats = {}

    for model_name in model_names:
        layers = results[model_name]['layers']
        metrics = get_metrics_by_category(layers)
        model_stats[model_name] = {
            'n_layers': len(layers),
            'input_pca': metrics['input_pca'],
            'output_pca': metrics['output_pca'],
            'mse': metrics['mse_normalized'],
        }

    # Output PCA variance by model
    pca_data = [model_stats[m]['output_pca'] for m in model_names]
    axes[0, 0].boxplot(pca_data, labels=model_names)
    axes[0, 0].set_ylabel('Explained Variance Ratio')
    axes[0, 0].set_title('Output PCA Variance by Model')
    axes[0, 0].grid(alpha=0.3, axis='y')
    plt.setp(axes[0, 0].xaxis.get_majorticklabels(), rotation=45, ha='right')

    # MSE by model
    mse_data = [model_stats[m]['mse'] for m in model_names]
    axes[0, 1].boxplot(mse_data, labels=model_names)
    axes[0, 1].set_ylabel('Normalized MSE')
    axes[0, 1].set_title('Autoencoder MSE by Model')
    axes[0, 1].set_yscale('log')
    axes[0, 1].grid(alpha=0.3, axis='y')
    plt.setp(axes[0, 1].xaxis.get_majorticklabels(), rotation=45, ha='right')

    # Layer count by model
    layer_counts = [model_stats[m]['n_layers'] for m in model_names]
    axes[1, 0].bar(model_names, layer_counts, color='steelblue', edgecolor='black')
    axes[1, 0].set_ylabel('Number of Layers')
    axes[1, 0].set_title('Layer Count by Model')
    axes[1, 0].grid(alpha=0.3, axis='y')
    for i, v in enumerate(layer_counts):
        axes[1, 0].text(i, v + 1, str(v), ha='center', fontweight='bold')
    plt.setp(axes[1, 0].xaxis.get_majorticklabels(), rotation=45, ha='right')

    # Average metrics table
    ax = axes[1, 1]
    ax.axis('off')

    table_data = [['Model', 'Layers', 'Avg Output PCA', 'Avg MSE']]
    for model_name in model_names:
        stats = model_stats[model_name]
        table_data.append([
            model_name,
            str(stats['n_layers']),
            f"{stats['output_pca'].mean():.4f}",
            f"{stats['mse'].mean():.6f}"
        ])

    table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                    colWidths=[0.3, 0.15, 0.25, 0.25])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.5)

    for i in range(len(table_data[0])):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')

    for i in range(1, len(table_data)):
        color = '#f0f0f0' if i % 2 == 0 else 'white'
        for j in range(len(table_data[0])):
            table[(i, j)].set_facecolor(color)

    plt.suptitle('Model Comparison', fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    file = output_dir / '00_aggregate_model_comparison.png'
    plt.savefig(file, dpi=150, bbox_inches='tight')
    print(f"  Saved: {file}")
    plt.close()


def create_aggregate_category_comparison(results: Dict, output_dir: Path):
    """Compare layer categories across all models."""
    print("\nGenerating aggregate category comparison...")

    categories = ['conv', 'dense', 'transformer', 'norm', 'pooling', 'other']
    colors = {'conv': 'blue', 'dense': 'green', 'transformer': 'red',
              'norm': 'orange', 'pooling': 'purple', 'other': 'gray'}

    # Collect all layers
    all_layers = []
    for model_name, data in results.items():
        all_layers.extend(data['layers'])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Layer Type Distribution Across All Models', fontsize=14, fontweight='bold')

    # Stacked bar chart of layer type counts per model
    model_names = sorted(results.keys())
    category_counts = {cat: [] for cat in categories}

    for model_name in model_names:
        layers = results[model_name]['layers']
        for cat in categories:
            count = len([l for l in layers if l['layer_category'] == cat])
            category_counts[cat].append(count)

    bottom = np.zeros(len(model_names))
    for cat in categories:
        counts = category_counts[cat]
        axes[0].bar(model_names, counts, label=cat, bottom=bottom, color=colors[cat], edgecolor='black')
        bottom += np.array(counts)

    axes[0].set_ylabel('Number of Layers')
    axes[0].set_title('Layer Type Composition by Model')
    axes[0].legend(loc='best')
    axes[0].grid(alpha=0.3, axis='y')
    plt.setp(axes[0].xaxis.get_majorticklabels(), rotation=45, ha='right')

    # MSE by category across all models
    for cat in categories:
        cat_layers = [l for l in all_layers if l['layer_category'] == cat]
        if cat_layers:
            mse_vals = np.array([l['train_mse_normalized'] for l in cat_layers])
            axes[1].scatter([cat] * len(mse_vals), mse_vals, alpha=0.5, s=50,
                          color=colors[cat], label=f"{cat} (n={len(mse_vals)})")

    axes[1].set_ylabel('Normalized MSE')
    axes[1].set_title('Autoencoder MSE by Layer Type (All Models)')
    axes[1].set_yscale('log')
    axes[1].grid(alpha=0.3, axis='y')
    axes[1].legend(loc='best', fontsize=9)
    plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=45, ha='right')

    plt.tight_layout()
    file = output_dir / '00_aggregate_category_comparison.png'
    plt.savefig(file, dpi=150, bbox_inches='tight')
    print(f"  Saved: {file}")
    plt.close()


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Analyze autoencoder layer profiles',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python analyze_autoencoder_profiles.py ./autoencoder_profiles_20260203_000534
  python analyze_autoencoder_profiles.py ./autoencoder_profiles_20260203_000534 --output results
        """)

    parser.add_argument('profile_dir', type=str,
                       help='Path to autoencoder profiles directory')
    parser.add_argument('--output', type=str, default=None,
                       help='Output directory for visualizations (default: <profile_dir>/analysis)')

    args = parser.parse_args()

    profile_dir = Path(args.profile_dir)
    if not profile_dir.exists():
        print(f"Error: Profile directory not found: {profile_dir}")
        return

    # Create output directory
    output_dir = Path(args.output) if args.output else profile_dir / 'analysis'
    output_dir.mkdir(exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Load all metrics
    print(f"\nLoading profiles from {profile_dir}...")
    results = load_profile_metrics(profile_dir)

    if not results:
        print("Error: No profile metrics found!")
        return

    # Analyze each model
    for model_name, data in results.items():
        print(f"\n{'='*80}")
        print(f"Analysis for {model_name}")
        print('='*80)

        layers = data['layers']

        # Print statistics
        print_statistics(layers)
        for category in ['conv', 'dense', 'transformer', 'norm', 'pooling', 'other']:
            cat_layers = [l for l in layers if l['layer_category'] == category]
            if cat_layers:
                print_statistics(cat_layers, category)

        # Create visualizations
        model_output_dir = output_dir / model_name
        model_output_dir.mkdir(exist_ok=True)

        create_histograms(layers, model_output_dir)
        create_scatterplots(layers, model_output_dir)
        create_category_histograms(layers, model_output_dir)
        create_category_scatterplots(layers, model_output_dir)
        create_layer_type_analysis(layers, model_output_dir)

        print(f"\n✓ Saved visualizations to {model_output_dir}")

    # Generate aggregate plots across all models
    if len(results) > 1:
        print(f"\n{'='*80}")
        print("Generating aggregate plots across all models")
        print('='*80)
        create_aggregate_histograms(results, output_dir)
        create_aggregate_scatterplots(results, output_dir)
        create_aggregate_model_comparison(results, output_dir)
        create_aggregate_category_comparison(results, output_dir)
        print(f"✓ Aggregate visualizations saved to {output_dir}")
    else:
        print(f"\nNote: Only one model found - skipping aggregate plots")

    print(f"\n{'='*80}")
    print(f"Analysis complete!")
    print(f"Results saved to: {output_dir}")
    print('='*80)


if __name__ == '__main__':
    main()
