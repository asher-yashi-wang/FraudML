"""Optional PyG GraphSAGE. Graphs must contain only entities observed at prediction time."""
from __future__ import annotations

def build_causal_bipartite_edges(frame, entity_cols, time_col="TransactionDT"):
    """Return transaction--entity edge table; caller must construct each split without future rows.
    Entity node IDs are namespaced by column to prevent collisions.

    ``transaction_index`` is the positional row index (0..n-1) in the
    time-sorted transaction frame. Build node features on
    ``frame.sort_values(time_col, kind="stable").reset_index(drop=True)``
    so the feature matrix preserves the same order for tied timestamps.
    Edges within each supplied frame have no temporal filtering. Use only a
    history snapshot plus the target transaction when scoring a target.
    """
    import pandas as pd
    # 排序后重置索引，让边引用局部节点行号。调用方需用相同顺序构造节点特征。
    ordered = frame.sort_values(time_col, kind="stable").reset_index(drop=True).copy()
    rows=[]
    # 实体键带列名前缀，避免两个字段恰好用了相同编码而连成同一节点。
    for col in entity_cols:
        if col in ordered:
            for i, v in ordered[col].items():
                if pd.notna(v): rows.append((i, f"{col}:{v}"))
    return pd.DataFrame(rows, columns=["transaction_index", "entity_key"])

def graphsage_model(in_channels, hidden_channels=64, out_channels=1):
    try:
        import torch.nn as nn
        from torch_geometric.nn import SAGEConv
    except ImportError as e: raise ImportError("Install PyTorch and PyG: pip install -e '.[gnn]'") from e
    class GraphSAGE(nn.Module):
        def __init__(self):
            super().__init__(); self.c1=SAGEConv(in_channels, hidden_channels); self.c2=SAGEConv(hidden_channels, out_channels)
        def forward(self, x, edge_index): return self.c2(self.c1(x, edge_index).relu(), edge_index)
    return GraphSAGE()
