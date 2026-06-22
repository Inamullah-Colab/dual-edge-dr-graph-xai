from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler, normalize


@dataclass
class GraphBuildResult:
    nodes: pd.DataFrame
    edges: pd.DataFrame
    graph: nx.Graph


class SimilarityGraphBuilder:
    """Builds an image-level kNN graph with decomposed X12/X34 edge weights."""

    def __init__(self, k: int = 8):
        self.k = k

    def build(self, meta: pd.DataFrame, x12: np.ndarray, x34: np.ndarray, fused: np.ndarray) -> GraphBuildResult:
        x12n = normalize(StandardScaler().fit_transform(x12))
        x34n = normalize(StandardScaler().fit_transform(x34))
        fusedn = normalize(StandardScaler().fit_transform(fused))
        nbrs = NearestNeighbors(n_neighbors=self.k + 1, metric="cosine", algorithm="brute").fit(fusedn)
        _, ind = nbrs.kneighbors(fusedn)
        rows, seen = [], set()
        for i in range(ind.shape[0]):
            for j in ind[i, 1:]:
                a, b = sorted((int(i), int(j)))
                if (a, b) in seen:
                    continue
                seen.add((a, b))
                x12_sim = float(np.dot(x12n[a], x12n[b]))
                x34_sim = float(np.dot(x34n[a], x34n[b]))
                rows.append({
                    "source_idx": a,
                    "target_idx": b,
                    "weight_fused": float(0.5 * x12_sim + 0.5 * x34_sim),
                    "weight_x12_spatial_cosine": x12_sim,
                    "weight_x34_jacobian_cosine": x34_sim,
                    "same_grade": int(meta.iloc[a].diagnosis == meta.iloc[b].diagnosis),
                })
        edges = pd.DataFrame(rows)
        graph = nx.Graph()
        for i, r in meta.reset_index(drop=True).iterrows():
            graph.add_node(int(i), id_code=str(r.id_code), source_id=str(r.source_id), diagnosis=int(r.diagnosis))
        for r in edges.itertuples(index=False):
            graph.add_edge(int(r.source_idx), int(r.target_idx), weight=float(r.weight_fused))
        nodes = self.node_metrics(meta, edges, graph)
        return GraphBuildResult(nodes=nodes, edges=edges, graph=graph)

    def node_metrics(self, meta: pd.DataFrame, edges: pd.DataFrame, graph: nx.Graph) -> pd.DataFrame:
        degree = dict(graph.degree())
        weighted = dict(graph.degree(weight="weight"))
        clustering = nx.clustering(graph, weight="weight")
        rows = []
        for i, r in meta.reset_index(drop=True).iterrows():
            inc = edges[(edges.source_idx == i) | (edges.target_idx == i)]
            rows.append({
                "id_code": r.id_code,
                "source_id": r.source_id,
                "diagnosis": int(r.diagnosis),
                "graph_degree": float(degree.get(i, 0)),
                "graph_weighted_degree": float(weighted.get(i, 0)),
                "graph_clustering": float(clustering.get(i, 0)),
                "graph_same_grade_neighbor_fraction": float(inc.same_grade.mean()) if not inc.empty else 0.0,
                "graph_mean_x12_similarity": float(inc.weight_x12_spatial_cosine.mean()) if not inc.empty else 0.0,
                "graph_mean_x34_similarity": float(inc.weight_x34_jacobian_cosine.mean()) if not inc.empty else 0.0,
                "graph_mean_fused_similarity": float(inc.weight_fused.mean()) if not inc.empty else 0.0,
            })
        return pd.DataFrame(rows)

    @staticmethod
    def save(result: GraphBuildResult, out_dir: str | Path, prefix: str = "graph") -> None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        result.nodes.to_csv(out / f"{prefix}_nodes.csv", index=False)
        result.edges.to_csv(out / f"{prefix}_edges.csv", index=False)
        nx.write_graphml(result.graph, out / f"{prefix}.graphml")
