#!/usr/bin/env python
"""GNN models for node-level and whole-KTN predictions."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    GCNConv,
    GATConv,
    GINConv,
    NNConv,
    global_mean_pool,
    global_add_pool,
    BatchNorm,
)






class MPBackbone(nn.Module):
    """Shared stack of graph convolutions, normalization, and residuals."""

    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        hidden_dim: int = 64,
        n_layers: int = 3,
        conv_type: str = "nnconv",
        dropout: float = 0.3,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.dropout = dropout

        self.input_mlp = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        for _ in range(n_layers):
            if conv_type == "nnconv":



                bottleneck = max(hidden_dim // 4, 8)
                edge_nn = nn.Sequential(
                    nn.Linear(edge_dim, bottleneck),
                    nn.ReLU(),
                    nn.Linear(bottleneck, hidden_dim * hidden_dim),
                )
                self.convs.append(NNConv(hidden_dim, hidden_dim, edge_nn,
                                         aggr="mean"))
            elif conv_type == "gat":
                n_heads = 4
                assert hidden_dim % n_heads == 0
                self.convs.append(GATConv(
                    hidden_dim, hidden_dim // n_heads,
                    heads=n_heads, edge_dim=edge_dim, dropout=dropout,
                ))
            elif conv_type == "gcn":
                self.convs.append(GCNConv(hidden_dim, hidden_dim))
            elif conv_type == "gin":
                mlp = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                )
                self.convs.append(GINConv(mlp))
            else:
                raise ValueError(f"Unknown conv_type: {conv_type}")

            self.bns.append(BatchNorm(hidden_dim))

    def forward(self, x, edge_index, edge_attr=None, batch=None):
        x = self.input_mlp(x)

        for conv, bn in zip(self.convs, self.bns):
            x_res = x
            if isinstance(conv, NNConv):
                x = conv(x, edge_index, edge_attr)
            elif isinstance(conv, GATConv):
                x = conv(x, edge_index, edge_attr=edge_attr)
            else:
                x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = x + x_res
        return x






class KTNNodeModel(nn.Module):
    """One prediction per node, with a sigmoid for the committor."""

    def __init__(
        self,
        node_dim: int = 6,
        edge_dim: int = 4,
        hidden_dim: int = 64,
        n_layers: int = 3,
        conv_type: str = "nnconv",
        dropout: float = 0.3,
        task: str = "committor",
    ):
        super().__init__()
        self.task = task
        self.backbone = MPBackbone(
            node_dim, edge_dim, hidden_dim, n_layers, conv_type, dropout,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, data):
        x = self.backbone(data.x, data.edge_index, data.edge_attr)
        out = self.head(x).squeeze(-1)
        if self.task == "committor":
            out = torch.sigmoid(out)
        return out






class KTNGraphModel(nn.Module):
    """Pool the node embeddings into whole-graph predictions."""

    def __init__(
        self,
        node_dim: int = 6,
        edge_dim: int = 4,
        hidden_dim: int = 64,
        n_layers: int = 3,
        conv_type: str = "nnconv",
        dropout: float = 0.3,
        readout: str = "mean",
        n_targets: int = 1,
    ):
        super().__init__()
        self.readout_type = readout
        self.backbone = MPBackbone(
            node_dim, edge_dim, hidden_dim, n_layers, conv_type, dropout,
        )
        self.output_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_targets),
        )

    def forward(self, data):
        x = self.backbone(data.x, data.edge_index, data.edge_attr)

        if self.readout_type == "sum":
            graph_emb = global_add_pool(x, data.batch)
        else:
            graph_emb = global_mean_pool(x, data.batch)

        return self.output_mlp(graph_emb)






class KTNMultiTaskModel(nn.Module):
    """Shared backbone with separate node and graph heads."""

    def __init__(
        self,
        node_dim: int = 6,
        edge_dim: int = 4,
        hidden_dim: int = 64,
        n_layers: int = 3,
        conv_type: str = "nnconv",
        dropout: float = 0.3,
        readout: str = "mean",
        n_graph_targets: int = 1,
        node_task: str = "committor",
    ):
        super().__init__()
        self.node_task = node_task
        self.readout_type = readout

        self.backbone = MPBackbone(
            node_dim, edge_dim, hidden_dim, n_layers, conv_type, dropout,
        )


        self.node_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )


        self.graph_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_graph_targets),
        )

    def forward(self, data):
        x = self.backbone(data.x, data.edge_index, data.edge_attr)


        node_out = self.node_head(x).squeeze(-1)
        if self.node_task == "committor":
            node_out = torch.sigmoid(node_out)


        if self.readout_type == "sum":
            graph_emb = global_add_pool(x, data.batch)
        else:
            graph_emb = global_mean_pool(x, data.batch)
        graph_out = self.graph_head(graph_emb)

        return node_out, graph_out
