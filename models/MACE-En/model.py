#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
简化的MACE-En模型实现
"""

import torch
import torch.nn as nn
from e3nn import o3
from e3nn.util.jit import compile_mode

from .modules.enhanced_radial import EnhancedBesselBasis
from ..MACE.modules.blocks import ScaleShiftBlock
from ..MACE.modules.tools.scatter import scatter_sum


@compile_mode("script")
class SimpleMACEEn(torch.nn.Module):
    """
    简化版MACE-En模型，专注于数值稳定性
    """
    
    def __init__(
        self,
        r_max: float,
        num_bessel: int,
        hidden_size: int,
        num_elements: int,
        atomic_inter_scale: float = 1.0,
        atomic_inter_shift: float = 0.0,
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.r_max = r_max
        
        # Enhanced radial embedding
        self.radial_embedding = EnhancedBesselBasis(
            r_max=r_max, 
            num_basis=num_bessel,
            eps=1e-8
        )
        
        # Simple node embedding
        self.node_embedding = nn.Linear(num_elements, hidden_size)
        
        # Message passing layers
        self.message_mlp = nn.Sequential(
            nn.Linear(num_bessel + hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        
        self.update_mlp = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.SiLU(), 
            nn.Linear(hidden_size, hidden_size),
        )
        
        # Energy prediction
        self.energy_mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.SiLU(),
            nn.Linear(hidden_size // 2, 1),
        )
        
        # Scale and shift
        self.scale_shift = ScaleShiftBlock(
            scale=atomic_inter_scale, 
            shift=atomic_inter_shift
        )
        
    def forward(self, data):
        """
        前向传播
        
        Args:
            data: 包含以下键的字典
                - node_attrs: [n_nodes, num_elements] 
                - positions: [n_nodes, 3]
                - edge_index: [2, n_edges]
                - batch: [n_nodes]
                - num_graphs: int
        """
        
        node_attrs = data["node_attrs"]
        positions = data["positions"] 
        edge_index = data["edge_index"]
        batch = data["batch"]
        num_graphs = data["num_graphs"]
        
        # 节点嵌入
        h = self.node_embedding(node_attrs)  # [n_nodes, hidden_size]
        
        # 计算边长度
        edge_vectors = positions[edge_index[1]] - positions[edge_index[0]]
        edge_lengths = torch.norm(edge_vectors, dim=-1, keepdim=True)  # [n_edges, 1]
        
        # 径向特征 (使用增强的数值稳定性)
        edge_attr = self.radial_embedding(edge_lengths)  # [n_edges, num_bessel]
        
        # 消息传递
        src, dst = edge_index
        h_src = h[src]  # [n_edges, hidden_size]
        
        # 组合节点和边特征
        message_input = torch.cat([h_src, edge_attr], dim=-1)  # [n_edges, hidden_size + num_bessel]
        messages = self.message_mlp(message_input)  # [n_edges, hidden_size]
        
        # 聚合消息
        h_agg = scatter_sum(messages, dst, dim=0, dim_size=h.size(0))  # [n_nodes, hidden_size]
        
        # 更新节点特征
        update_input = torch.cat([h, h_agg], dim=-1)  # [n_nodes, hidden_size * 2]
        h = self.update_mlp(update_input)  # [n_nodes, hidden_size]
        
        # 预测节点能量
        node_energies = self.energy_mlp(h).squeeze(-1)  # [n_nodes]
        
        # 聚合为图能量
        graph_energies = scatter_sum(
            node_energies, batch, dim=0, dim_size=num_graphs
        )  # [num_graphs]
        
        # 应用scale和shift
        graph_energies = self.scale_shift(graph_energies)
        
        return {"energy": graph_energies}


# 为了兼容性，创建别名
ScaleShiftMACEEn = SimpleMACEEn
import torch
from e3nn import o3
from e3nn.util.jit import compile_mode

# Import from original MACE
from ..MACE.modules.tools.scatter import scatter_sum
from ..MACE.modules.blocks import (
    EquivariantProductBasisBlock,
    InteractionBlock,
    LinearNodeEmbeddingBlock,
    LinearReadoutBlock,
    NonLinearReadoutBlock,
    PolynomialCutoff,
)
from ..MACE.modules.utils import (
    get_edge_vectors_and_lengths,
    get_outputs,
    get_symmetric_displacement,
)

# Import enhanced radial basis
from .modules.enhanced_radial import EnhancedBesselBasis, HybridBasis, AdaptiveBesselBasis


@compile_mode("script")
class EnhancedRadialEmbeddingBlock(torch.nn.Module):
    """
    Enhanced Radial Embedding with improved numerical stability
    """
    def __init__(
        self,
        r_max: float,
        num_bessel: int,
        num_polynomial_cutoff: int,
        radial_type: str = "enhanced_bessel",  # New options
        eps: float = 1e-8,
    ):
        super().__init__()
        
        # Choose radial basis type
        if radial_type == "enhanced_bessel":
            self.radial_fn = EnhancedBesselBasis(r_max=r_max, num_basis=num_bessel, eps=eps)
        elif radial_type == "hybrid":
            self.radial_fn = HybridBasis(r_max=r_max, num_bessel=num_bessel//2, 
                                       num_gaussian=num_bessel//2, eps=eps)
        elif radial_type == "adaptive":
            self.radial_fn = AdaptiveBesselBasis(r_max=r_max, num_basis=num_bessel, eps=eps)
        else:
            # Fallback to enhanced bessel
            self.radial_fn = EnhancedBesselBasis(r_max=r_max, num_basis=num_bessel, eps=eps)
            
        self.cutoff_fn = PolynomialCutoff(r_max=r_max, p=num_polynomial_cutoff)
        self.out_dim = getattr(self.radial_fn, 'out_dim', num_bessel)

    def forward(self, edge_lengths: torch.Tensor):
        radial = self.radial_fn(edge_lengths)  # [n_edges, n_basis]
        cutoff = self.cutoff_fn(edge_lengths)  # [n_edges, 1]
        return radial * cutoff  # [n_edges, n_basis]


@compile_mode("script")
class MACE_En(torch.nn.Module):
    """
    MACE-En: Enhanced MACE with improved numerical stability
    """
    def __init__(
        self,
        r_max: float,
        num_bessel: int,
        num_polynomial_cutoff: int,
        max_ell: int,
        interaction_cls: Type[InteractionBlock],
        interaction_cls_first: Type[InteractionBlock],
        num_interactions: int,
        num_elements: int,
        hidden_irreps: o3.Irreps,
        MLP_irreps: o3.Irreps,
        atomic_energies: np.ndarray,
        avg_num_neighbors: float,
        atomic_numbers: List[int],
        correlation: int,
        gate: Optional[Callable],
        radial_MLP: Optional[List[int]] = None,
        radial_type: str = "enhanced_bessel",  # New parameter
        numerical_eps: float = 1e-8,  # Numerical stability parameter
    ):
        super().__init__()
        
        self.numerical_eps = numerical_eps
        
        self.register_buffer(
            "atomic_numbers", torch.tensor(atomic_numbers, dtype=torch.int64)
        )
        self.register_buffer(
            "r_max", torch.tensor(r_max, dtype=torch.get_default_dtype())
        )
        self.register_buffer(
            "num_interactions", torch.tensor(num_interactions, dtype=torch.int64)
        )
        
        # Enhanced embedding
        node_attr_irreps = o3.Irreps([(num_elements, (0, 1))])
        node_feats_irreps = o3.Irreps([(hidden_irreps.count(o3.Irrep(0, 1)), (0, 1))])
        
        self.node_embedding = LinearNodeEmbeddingBlock(
            irreps_in=node_attr_irreps, irreps_out=node_feats_irreps
        )
        
        # Enhanced radial embedding with improved stability
        self.radial_embedding = EnhancedRadialEmbeddingBlock(
            r_max=r_max,
            num_bessel=num_bessel,
            num_polynomial_cutoff=num_polynomial_cutoff,
            radial_type=radial_type,
            eps=numerical_eps,
        )
        
        edge_feats_irreps = o3.Irreps(f"{self.radial_embedding.out_dim}x0e")

        # Spherical harmonics with stability
        sh_irreps = o3.Irreps.spherical_harmonics(max_ell)
        num_features = hidden_irreps.count(o3.Irrep(0, 1))
        interaction_irreps = (sh_irreps * num_features).sort()[0].simplify()
        
        self.spherical_harmonics = o3.SphericalHarmonics(
            sh_irreps, normalize=True, normalization="component"
        )
        
        if radial_MLP is None:
            radial_MLP = [64, 64, 64]

        # Enhanced atomic energies with stability
        self.atomic_energies_fn = torch.nn.Linear(num_elements, 1)
        
        # Apply Xavier/Glorot initialization for stability
        torch.nn.init.xavier_uniform_(self.atomic_energies_fn.weight)
        torch.nn.init.zeros_(self.atomic_energies_fn.bias)

        # Build interaction layers
        inter = interaction_cls_first(
            node_attrs_irreps=node_attr_irreps,
            node_feats_irreps=node_feats_irreps,
            edge_attrs_irreps=sh_irreps,
            edge_feats_irreps=edge_feats_irreps,
            target_irreps=interaction_irreps,
            hidden_irreps=hidden_irreps,
            avg_num_neighbors=avg_num_neighbors,
            radial_MLP=radial_MLP,
        )
        self.interactions = torch.nn.ModuleList([inter])

        use_sc_first = False
        if "Residual" in str(interaction_cls_first):
            use_sc_first = True

        node_feats_irreps_out = inter.target_irreps
        prod = EquivariantProductBasisBlock(
            node_feats_irreps=node_feats_irreps_out,
            target_irreps=hidden_irreps,
            correlation=correlation,
            num_elements=num_elements,
            use_sc=use_sc_first,
        )
        self.products = torch.nn.ModuleList([prod])
        self.readouts = torch.nn.ModuleList()
        self.readouts.append(LinearReadoutBlock(hidden_irreps))

        # Build remaining interaction layers
        for i in range(num_interactions - 1):
            if i == num_interactions - 2:
                hidden_irreps_out = str(hidden_irreps[0])
            else:
                hidden_irreps_out = hidden_irreps
                
            inter = interaction_cls(
                node_attrs_irreps=node_attr_irreps,
                node_feats_irreps=hidden_irreps,
                edge_attrs_irreps=sh_irreps,
                edge_feats_irreps=edge_feats_irreps,
                target_irreps=interaction_irreps,
                hidden_irreps=hidden_irreps_out,
                avg_num_neighbors=avg_num_neighbors,
                radial_MLP=radial_MLP,
            )
            self.interactions.append(inter)
            
            prod = EquivariantProductBasisBlock(
                node_feats_irreps=interaction_irreps,
                target_irreps=hidden_irreps_out,
                correlation=correlation,
                num_elements=num_elements,
                use_sc=True,
            )
            self.products.append(prod)
            
            if i == num_interactions - 2:
                self.readouts.append(
                    NonLinearReadoutBlock(hidden_irreps_out, MLP_irreps, gate)
                )
            else:
                self.readouts.append(LinearReadoutBlock(hidden_irreps))

    def forward(
        self,
        data: Dict[str, torch.Tensor],
        training: bool = False,
        compute_force: bool = True,
        compute_virials: bool = False,
        compute_stress: bool = False,
        compute_displacement: bool = False,
    ) -> Dict[str, Optional[torch.Tensor]]:
        # Setup with numerical stability
        data["node_attrs"].requires_grad_(True)
        data["positions"].requires_grad_(True)
        
        # Handle different batch size specifications
        if 'num_graphs' in data:
            num_graphs = data['num_graphs']
        else:
            num_graphs = data["ptr"].numel() - 1
            
        displacement = torch.zeros(
            (num_graphs, 3, 3),
            dtype=data["positions"].dtype,
            device=data["positions"].device,
        )

        if compute_virials or compute_stress or compute_displacement:
            (
                data["positions"],
                data["shifts"],
                displacement,
            ) = get_symmetric_displacement(
                positions=data["positions"],
                unit_shifts=data["unit_shifts"],
                cell=data["cell"],
                edge_index=data["edge_index"],
                num_graphs=num_graphs,
                batch=data["batch"],
            )

        # Atomic energies with numerical stability
        node_e0 = self.atomic_energies_fn(data["node_attrs"])
        e0 = scatter_sum(
            src=node_e0, index=data["batch"], dim=-1, dim_size=num_graphs
        )

        # Enhanced embeddings
        node_feats = self.node_embedding(data["node_attrs"])
        vectors, lengths = get_edge_vectors_and_lengths(
            positions=data["positions"],
            edge_index=data["edge_index"],
            shifts=data["shifts"],
        )
        
        # Add small epsilon to lengths for stability
        lengths = torch.clamp(lengths, min=self.numerical_eps)
        
        edge_attrs = self.spherical_harmonics(vectors)
        edge_feats = self.radial_embedding(lengths)

        # Enhanced interactions with stability checks
        energies = [e0]
        node_energies_list = [node_e0]
        node_feats_list = []
        
        for interaction, product, readout in zip(
            self.interactions, self.products, self.readouts
        ):
            node_feats, sc = interaction(
                node_attrs=data["node_attrs"],
                node_feats=node_feats,
                edge_attrs=edge_attrs,
                edge_feats=edge_feats,
                edge_index=data["edge_index"],
            )
            
            node_feats = product(
                node_feats=node_feats,
                sc=sc,
                node_attrs=data["node_attrs"],
            )
            
            node_feats_list.append(node_feats)
            node_energies = readout(node_feats).squeeze(-1)
            
            # Numerical stability check
            if torch.isnan(node_energies).any() or torch.isinf(node_energies).any():
                print("Warning: NaN or Inf detected in node energies, applying clipping")
                node_energies = torch.clamp(node_energies, min=-100, max=100)
            
            energy = scatter_sum(
                src=node_energies, index=data["batch"], dim=-1, dim_size=num_graphs
            )
            energies.append(energy)
            node_energies_list.append(node_energies)

        # Concatenate node features
        node_feats_out = torch.cat(node_feats_list, dim=-1)

        # Sum energy contributions with stability
        contributions = torch.stack(energies, dim=-1)
        total_energy = torch.sum(contributions, dim=-1)
        
        # Final stability check
        if torch.isnan(total_energy).any() or torch.isinf(total_energy).any():
            print("Warning: NaN or Inf in total energy, applying emergency clipping")
            total_energy = torch.clamp(total_energy, min=-1000, max=1000)
        
        node_energy_contributions = torch.stack(node_energies_list, dim=-1)
        node_energy = torch.sum(node_energy_contributions, dim=-1)

        # Outputs
        forces, virials, stress = get_outputs(
            energy=total_energy,
            positions=data["positions"],
            displacement=displacement,
            cell=data["cell"],
            training=training,
            compute_force=compute_force,
            compute_virials=compute_virials,
            compute_stress=compute_stress,
        )

        return {
            "energy": total_energy,
            "node_energy": node_energy,
            "contributions": contributions,
            "forces": forces,
            "virials": virials,
            "stress": stress,
            "displacement": displacement,
            "node_feats": node_feats_out,
        }
