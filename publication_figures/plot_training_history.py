#!/usr/bin/env python3
"""
Plot training history (train and validation loss) for all patch variants.
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Define patch variants (organized as row, col positions)
variant_positions = [
    ('VJJ7744', 0, 0),
    ('SHX8459', 0, 1),
    ('VJJ7744_nohomo', 1, 0),
    ('SHX8459_nohomo', 1, 1),
    ('VJJ7744_notv', 2, 0),
    ('SHX8459_notv', 2, 1),
    ('VJJ7744_notv_nohomo', 3, 0),
    ('SHX8459_notv_nohomo', 3, 1)
]

# Base directory
base_dir = Path('patch_variants_20260101_001131')

# Create labels for variants (no PII)
def get_variant_label(variant):
    """Create clean variant label without PII"""
    # Determine attack type
    if variant.startswith('VJJ7744'):
        attack_type = 'Impersonation'
    else:
        attack_type = 'Disruption'

    # Determine configuration
    if '_notv' in variant and '_nohomo' in variant:
        config = 'No TV + No Homo'
    elif '_notv' in variant:
        config = 'No TV Loss'
    elif '_nohomo' in variant:
        config = 'No Homography'
    else:
        config = 'Full (TV + Homo)'

    return f'{attack_type}\n{config}'

# Create figure with subplots (4 rows x 2 columns)
fig, axes = plt.subplots(4, 2, figsize=(14, 16))

for variant, row, col in variant_positions:
    ax = axes[row, col]

    # Load training history
    history_file = base_dir / f'patches_{variant}' / 'training_history.csv'

    if not history_file.exists():
        ax.text(0.5, 0.5, f'No data', ha='center', va='center',
               transform=ax.transAxes, fontsize=10)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        continue

    df = pd.read_csv(history_file)

    # Plot training and validation loss
    ax.plot(df['epoch'], df['loss'], 'b-', linewidth=2, label='Train Loss', alpha=0.8)
    ax.plot(df['epoch'], df['val_score'], 'r-', linewidth=2, label='Val Loss', alpha=0.8)

    ax.set_xlabel('Epoch', fontsize=10)
    ax.set_ylabel('Loss', fontsize=10)
    ax.set_title(get_variant_label(variant), fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=9)

    # Add learning rate change marker if applicable
    lr_change_epochs = df[df['learning_rate'] != df['learning_rate'].iloc[0]]['epoch']
    if len(lr_change_epochs) > 0:
        first_lr_change = lr_change_epochs.iloc[0]
        ax.axvline(x=first_lr_change, color='gray', linestyle='--',
                  linewidth=1, alpha=0.5, label=f'LR change (epoch {first_lr_change})')

# Add column labels at the top
column_labels = ['IMPERSONATION ATTACK', 'DISRUPTION ATTACK']
for j, label in enumerate(column_labels):
    axes[0, j].text(0.5, 1.15, label, ha='center', va='bottom',
                   transform=axes[0, j].transAxes, fontsize=13, fontweight='bold')

plt.suptitle('Training History - Patch Variants Ablation Study',
             fontsize=14, fontweight='bold', y=0.995)
plt.tight_layout(rect=[0, 0, 1, 0.99])
plt.savefig('training_history_all_variants.png', dpi=300, bbox_inches='tight')
print("Training history plot saved to training_history_all_variants.png")

# Print summary statistics
print("\n" + "="*80)
print("TRAINING HISTORY SUMMARY")
print("="*80)

for variant, row, col in variant_positions:
    history_file = base_dir / f'patches_{variant}' / 'training_history.csv'

    if not history_file.exists():
        continue

    df = pd.read_csv(history_file)

    # Get clean label
    label = get_variant_label(variant).replace('\n', ' ')

    final_train_loss = df['loss'].iloc[-1]
    final_val_loss = df['val_score'].iloc[-1]
    best_val_loss = df['val_score'].min()
    best_val_epoch = df.loc[df['val_score'].idxmin(), 'epoch']

    print(f"\n{label}:")
    print(f"  Final train loss: {final_train_loss:.4f}")
    print(f"  Final val loss:   {final_val_loss:.4f}")
    print(f"  Best val loss:    {best_val_loss:.4f} (epoch {best_val_epoch:.0f})")
    print(f"  Total epochs:     {len(df)}")

print("\n" + "="*80)
