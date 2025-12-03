```python
def patch_reg_loss(self):
    patch = self.patch  # shape (C, 256, 128)
    C, H, W = patch.shape
    
    tv_h = torch.pow(patch[:, :, 1:] - patch[:, :, :-1], 2).sum()
    tv_v = torch.pow(patch[:, 1:, :] - patch[:, :-1, :], 2).sum()
    
    # Number of comparisons: C × (256×127 + 255×128) = C × 65152
    num_comparisons = C * (H * (W - 1) + (H - 1) * W)
    
    loss = (tv_h + tv_v) / num_comparisons
    
    # For [-1, 1] range, mean squared diff is typically 0.01-0.04 for natural images
    # Scale by 2-3x to get 0-0.1 range
    loss = loss * 2.5
    
    return loss
```

https://claude.ai/chat/17d19287-27cf-4053-9e7b-8249d67188de

Patch regularization loss, missing from currently committed file. Ranges 0-0.1, directly added to loss term.
