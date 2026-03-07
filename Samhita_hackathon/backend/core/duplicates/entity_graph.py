import pandas as pd
import numpy as np
from rapidfuzz import fuzz


def build_entity_graph(df, key_columns=None, similarity_threshold=80, max_comparisons=50000):
    """
    Build a similarity graph and cluster duplicate entities.
    
    Uses RapidFuzz for pairwise string similarity and connected components
    for clustering (without NetworkX dependency — uses simple union-find).
    
    Args:
        df: Input DataFrame
        key_columns: Columns to compare. If None, auto-detects string columns.
        similarity_threshold: Minimum similarity score (0-100) to consider a match
        max_comparisons: Maximum pairwise comparisons to limit computation
    
    Returns:
        dict with clusters, stats, and edges
    """
    if key_columns is None:
        key_columns = _auto_detect_key_columns(df)
    
    if not key_columns:
        return {"clusters": [], "total_clusters": 0, "edges": [], "message": "No suitable columns for entity resolution"}
    
    n = len(df)
    edges = []
    
    # Create concatenated comparison strings
    compare_values = []
    for _, row in df[key_columns].iterrows():
        val = " ".join(str(v).lower().strip() for v in row.values if pd.notnull(v) and str(v).strip())
        compare_values.append(val)
    
    # Pairwise comparison with early stopping
    comparison_count = 0
    for i in range(n):
        if not compare_values[i]:
            continue
        for j in range(i + 1, min(i + 100, n)):  # Window-based comparison
            if not compare_values[j]:
                continue
            
            score = fuzz.token_sort_ratio(compare_values[i], compare_values[j])
            if score >= similarity_threshold:
                edges.append((i, j, score))
            
            comparison_count += 1
            if comparison_count >= max_comparisons:
                break
        
        if comparison_count >= max_comparisons:
            break
    
    # Build clusters using Union-Find
    parent = list(range(n))
    
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    
    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py
    
    for i, j, _ in edges:
        union(i, j)
    
    # Group into clusters
    from collections import defaultdict
    cluster_map = defaultdict(list)
    for idx in range(n):
        root = find(idx)
        cluster_map[root].append(idx)
    
    # Filter to only clusters with 2+ members
    clusters = []
    for members in cluster_map.values():
        if len(members) >= 2:
            cluster_records = []
            for idx in members:
                record = df.iloc[idx].to_dict()
                record["_row_index"] = int(idx)
                # Convert non-serializable types
                for k, v in record.items():
                    if isinstance(v, (pd.Timestamp, np.integer, np.floating)):
                        record[k] = str(v)
                    elif pd.isnull(v):
                        record[k] = None
                cluster_records.append(record)
            clusters.append({
                "cluster_id": len(clusters) + 1,
                "size": len(members),
                "records": cluster_records,
            })
    
    return {
        "clusters": clusters,
        "total_clusters": len(clusters),
        "total_edges": len(edges),
        "total_records_in_clusters": sum(c["size"] for c in clusters),
        "columns_used": key_columns,
    }


def _auto_detect_key_columns(df):
    """Auto-detect columns suitable for entity matching (string columns with moderate cardinality)."""
    candidates = []
    for col in df.columns:
        if df[col].dtype == 'object':
            cardinality_ratio = df[col].nunique() / len(df) if len(df) > 0 else 0
            if 0.1 < cardinality_ratio < 0.95:  # Not constant, not unique identifiers
                candidates.append(col)
    
    # Prioritize name/email/phone-like columns
    priority_keywords = ['name', 'email', 'phone', 'address', 'company']
    prioritized = [c for c in candidates if any(kw in c.lower() for kw in priority_keywords)]
    
    if prioritized:
        return prioritized[:5]
    return candidates[:5]
