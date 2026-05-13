import torch
import torch.nn as nn
import torch.nn.functional as F

class ShiftBN(nn.BatchNorm2d):
    """Multiplier-Free Batch Normalization. Weights are snapped to powers of 2."""
    def forward(self, x):
        if self.weight is not None:
            # Snap gamma to exact power of 2
            sw = torch.sign(self.weight) * (2.0 ** torch.round(torch.log2(self.weight.abs().clamp(1e-6))))
        else:
            sw = None
            
        return F.batch_norm(
            x, self.running_mean, self.running_var, 
            sw, self.bias, 
            self.training or not self.track_running_stats, 
            self.momentum, self.eps
        )
