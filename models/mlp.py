import torch
import torch.nn as nn
from typing import List, Optional, Union


class PowerEstimatorMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int = 1,
        depth = 2,
        dropout: float = 0.,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        depth = 1
        dims = [input_dim] + [36] * depth

        self.blocks = nn.ModuleList()
        
        if depth > 0:
            for i in range(depth):
                in_dim, out_dim = dims[i], dims[i + 1]
                block = [
                    nn.Linear(in_dim, out_dim, bias=True, dtype=dtype),
                    nn.LayerNorm(out_dim, elementwise_affine=True, dtype=dtype),
                    nn.ReLU(),
                ]
                if dropout > 0:
                    block.append(nn.Dropout(dropout))
                self.blocks.append(nn.Sequential(*block))
        # 输出头
        self.head = nn.Linear(dims[-1], output_dim, bias=True, dtype=dtype)
        setattr(self.head, "_is_output_head", True) # set for initialize checker

        # 权重初始化
        self.apply(lambda m: _init_weights(m, scheme="xavier", activation="tanh"))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        pred = self.head(x)
        return pred

    def num_parameters(self, trainable_only: bool = True) -> int:
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())

def _init_weights(m: nn.Module,
                  scheme: str = "xavier",
                  activation: str = "tanh"):
    """
    Initialize layers with a sensible default based on activation.
    - scheme="auto": choose Xavier for tanh/silu/gelu, Kaiming for relu/leaky_relu; orthogonal for RNN-like not used here.
    - scheme in {"xavier","kaiming","orthogonal","normal"}: force that scheme for hidden Linear layers.
    The output head is always small-std normal.
    """
    if isinstance(m, nn.Linear):
        is_head = getattr(m, "_is_output_head", False)

        if is_head:
            # Small init for regression stability
            nn.init.normal_(m.weight, mean=0.0, std=1e-2)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
            return

        act = activation.lower() if isinstance(activation, str) else "tanh"

        if scheme == "auto":
            if act in ("relu", "leaky_relu"):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            elif act in ("gelu", "silu", "swish", "tanh"):
                gain = nn.init.calculate_gain("tanh" if act == "tanh" else "linear")
                # GELU/SILU 不有闭式增益，使用接近线性的增益更稳定
                nn.init.xavier_normal_(m.weight, gain=gain)
            else:
                nn.init.xavier_normal_(m.weight)
        elif scheme == "xavier":
            nn.init.xavier_normal_(m.weight)
        elif scheme == "kaiming":
            nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
        elif scheme == "orthogonal":
            nn.init.orthogonal_(m.weight, gain=1.0)
        elif scheme == "normal":
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
        else:
            nn.init.xavier_normal_(m.weight)

        if m.bias is not None:
            nn.init.zeros_(m.bias)
