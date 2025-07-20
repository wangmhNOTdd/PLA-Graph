import torch
from torch import nn
from torch.nn import functional as F

def swiglu(gate, x):
    """
    Hand-written SwiGLU activation function.
    SwiGLU(gate, x) = Swish(gate) * x = (gate * sigmoid(gate)) * x
    
    Args:
        gate: Gate values tensor
        x: Input values tensor
    
    Returns:
        SwiGLU activated tensor
    """
    return F.silu(gate) * x  # F.silu is the Swish activation function


class SmallMLP(nn.Module):
    def __init__(
        self,
        in_dim,
        inter_dim,
        out_dim,
        dropout_p=0.0,
        num_layers=2,
        use_ln=False,
    ):
        super().__init__()

        if num_layers == 1:
            self.mlp = nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.Dropout(p=dropout_p) if dropout_p > 0 else nn.Identity(),
                nn.Mish(),
            )
        else:
            mlp_layers = []
            for i in range(num_layers):
                if i == 0:
                    mlp_layers.append(nn.Linear(in_dim, inter_dim))
                    if use_ln:
                        mlp_layers.append(nn.LayerNorm(inter_dim))
                    mlp_layers.append(nn.Mish())
                elif i != num_layers - 1:
                    mlp_layers.append(nn.Linear(inter_dim, inter_dim))
                    if use_ln:
                        mlp_layers.append(nn.LayerNorm(inter_dim))
                    mlp_layers.append(nn.Mish())
                else:
                    mlp_layers.append(nn.Linear(inter_dim, out_dim))

                if dropout_p > 0:
                    mlp_layers.append(nn.Dropout(p=dropout_p))

            self.mlp = nn.Sequential(*mlp_layers)


    def forward(self, x):
        return self.mlp(x)
    

class GatedMLPSingle(nn.Module):
    def __init__(
        self,
        in_dim,
        inter_dim,
        out_dim,
        dropout_p=0.0,
        use_ln=False,
    ):
        super().__init__()

        # Uncomment if you want dropout here
        # self.dropout_p = dropout_p

        self.fc1 = nn.Linear(in_dim, 2 * inter_dim, bias=True)
        self.fc2 = nn.Linear(inter_dim, out_dim, bias=True)
        self.use_ln = use_ln

        if self.use_ln:
            self.ln = nn.LayerNorm(2 * inter_dim, eps=1e-8)

        # if dropout_p > 0:
        #     self.dropout = nn.Dropout(p=dropout_p)


    def forward(self, x):
        if self.use_ln:
            y = self.ln(self.fc1(x))
        else:
            y = self.fc1(x)

        y, gate = y.chunk(2, dim=-1)
        y = swiglu(gate, y)

        # if self.dropout_p > 0:
        #     y = self.dropout(y)
        y = self.fc2(y)
        
        return y
    

class GatedMLPMulti(nn.Module):
    def __init__(
        self,
        in_dim,
        inter_dim,
        out_dim,
        dropout_p=0.0,
        num_layers=2,
        use_ln=False,
    ):
        super().__init__()

        if num_layers == 1:
            self.mlp = nn.Sequential(
                GatedMLPSingle(in_dim, inter_dim, out_dim, dropout_p=dropout_p, use_ln=False),
                nn.Dropout(p=dropout_p) if dropout_p > 0 else nn.Identity(),
                nn.Mish(),
            )
        else:
            mlp_layers = []
            for i in range(num_layers):
                if i == 0:
                    mlp_layers.append(GatedMLPSingle(in_dim, inter_dim, inter_dim, dropout_p=dropout_p, use_ln=use_ln))
                elif i != num_layers - 1:
                    mlp_layers.append(GatedMLPSingle(inter_dim, inter_dim, inter_dim, dropout_p=dropout_p, use_ln=use_ln))
                else:
                    mlp_layers.append(GatedMLPSingle(inter_dim, inter_dim, out_dim, dropout_p=dropout_p, use_ln=use_ln))
                
                if dropout_p > 0:
                    mlp_layers.append(nn.Dropout(p=dropout_p))

                mlp_layers.append(nn.Mish())

            self.mlp = nn.Sequential(*mlp_layers)


    def forward(self, x):
        return self.mlp(x)