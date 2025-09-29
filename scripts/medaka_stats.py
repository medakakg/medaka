"""
This script computes key graph statistics from the MEDAKA triples CSV.

Expected CSV columns:
- Subject, Relation, Object, Filename, Count, Confidence

What it reports:
- Node and edge counts
- Predicate/Relation-wise triple and unique object counts
- Degree distribution (avg/min/max)
- Betweenness centrality
- Degree assortativity coefficient
- Top drugs by total degree
"""

# ------------------ Imports ------------------ #
import argparse
import pandas as pd
import networkx as nx
import sys

# ------------------ Helpers ------------------ #
def _clean_relation_to_type(rel: str) -> str:
    """
    Convert a relation string (e.g., 'hassideeffect', 'hasWarning')
    into a readable node type label (e.g., 'Sideeffect', 'Warning').
    """
    if not isinstance(rel, str):
        return "Object"
    r = rel.strip()
    # remove leading 'has' (case-insensitive)
    rl = r.lower()
    if rl.startswith("has"):
        r = r[3:]
    # title-case the remainder
    return r.strip().title() or "Object"

def _validate_columns(df: pd.DataFrame):
    required = {"Subject", "Relation", "Object"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Input CSV is missing required columns: {sorted(missing)}. "
            f"Found columns: {list(df.columns)}"
        )

def build_graph(df: pd.DataFrame) -> nx.MultiDiGraph:
    """
    Constructs a directed multigraph from the input dataframe.
    Each node is labeled with a type, and each edge represents a relation.
    
    Parameters:
        df (pd.DataFrame): Columns: Subject, Relation, Object
    Returns:
        nx.MultiDiGraph: The MEDAKA KG
    """
    G = nx.MultiDiGraph()
    for _, row in df.iterrows():
        subj, rel, obj = row["Subject"], row["Relation"], row["Object"]
        G.add_node(subj, node_type="Drug")
        G.add_node(obj, node_type=_clean_relation_to_type(rel))
        # keep edge attribute name 'predicate' for compatibility with prior code
        G.add_edge(subj, obj, predicate=rel)
    return G

def basic_stats(df: pd.DataFrame, G: nx.MultiDiGraph):
    """Prints basic statistics: total triples, node count, unique drugs and relations."""
    print(f"Total subject-relation-object triples : {G.number_of_edges()}")
    print(f"Total unique nodes: {G.number_of_nodes()}")
    print(f"Unique drugs : {df['Subject'].nunique()}")
    print(f"Unique relations (edge types): {df['Relation'].nunique()}")

def predicate_summary(df: pd.DataFrame):
    """
    Prints detailed statistics for each relation:
    number of triples, unique object count, and average per drug.
    """
    print("\nRelation-wise statistics:")
    relation_counts = df["Relation"].value_counts()
    num_drugs = df["Subject"].nunique()
    for rel, count in relation_counts.items():
        unique_objs = df[df["Relation"] == rel]["Object"].nunique()
        avg_per_drug = count / num_drugs if num_drugs else 0.0
        print(f"{rel}: {count} triples, {unique_objs} unique objects, avg per drug : {avg_per_drug:.2f}")

def degree(G_undirected: nx.Graph):
    """Computes and prints degree stats from the undirected graph."""
    degree_sequence = [d for _, d in G_undirected.degree()]
    if not degree_sequence:
        print("\nAverage degree: 0.00")
        print("Max degree: 0, Min degree: 0")
        return
    print(f"\nAverage degree: {sum(degree_sequence) / len(degree_sequence):.2f}")
    print(f"Max degree: {max(degree_sequence)}, Min degree: {min(degree_sequence)}")

def centrality(G_undirected: nx.Graph):
    """Calculates betweenness centrality and prints top 5 nodes + summary."""
    print("\nBetweenness centrality (top 5 nodes):")
    if G_undirected.number_of_nodes() == 0:
        print("  (graph is empty)")
        print("Avg: 0.0000, Min: 0.0000, Max: 0.0000")
        return
    bc = nx.betweenness_centrality(G_undirected)
    if not bc:
        print("  (no centrality values)")
        print("Avg: 0.0000, Min: 0.0000, Max: 0.0000")
        return
    top = sorted(bc.items(), key=lambda x: x[1], reverse=True)[:5]
    for node, score in top:
        print(f"  {node}: {score:.4f}")
    print(f"Avg: {sum(bc.values())/len(bc):.4f}, Min: {min(bc.values()):.4f}, Max: {max(bc.values()):.4f}")

def assortativity(G_undirected: nx.Graph):
    """Computes the degree assortativity coefficient (if possible)."""
    print("\nDegree assortativity:")
    try:
        if G_undirected.number_of_edges() == 0:
            print("Coefficient: 0.0000 (graph has no edges)")
            return
        assort = nx.degree_assortativity_coefficient(G_undirected)
        print(f"Coefficient: {assort:.4f}")
    except Exception as e:
        print("Could not compute assortativity.", e)

def top_drugs(G: nx.MultiDiGraph):
    """Identifies the top 5 drug nodes by total degree."""
    drug_nodes = [n for n, d in G.nodes(data=True) if d.get("node_type") == "Drug"]
    drug_degrees = {n: G.degree(n) for n in drug_nodes} if drug_nodes else {}
    topk = sorted(drug_degrees.items(), key=lambda x: x[1], reverse=True)[:5]
    print("\nTop 5 drugs by degree:")
    if not topk:
        print("  (no drug nodes)")
        return
    for drug, deg in topk:
        print(f"  {drug}: {deg} connections")

# ------------------ Main ------------------ #
def main(input_csv: str):
    """Main function to compute graph statistics from MEDAKA CSV."""
    df = pd.read_csv(input_csv)
    # be forgiving about column casing/whitespace
    df.columns = [c.strip() for c in df.columns]
    _validate_columns(df)

    G = build_graph(df)
    G_undirected = G.to_undirected()

    basic_stats(df, G)
    predicate_summary(df)
    degree(G_undirected)
    centrality(G_undirected)
    assortativity(G_undirected)
    top_drugs(G)

# ------------------ CLI ------------------ #
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute structural statistics of MEDAKA from triples CSV.")
    parser.add_argument("--input-csv", help="Path to MEDAKA CSV (columns: Subject, Relation, Object, ...).")
    # Backward-compatible alias:
    parser.add_argument("--input", help="(Alias) Path to MEDAKA CSV.", dest="input_alias")
    args = parser.parse_args()

    input_path = args.input_csv or args.input_alias
    if not input_path:
        print("ERROR: Please provide --input-csv <path> (or --input <path>).", file=sys.stderr)
        sys.exit(2)

    main(input_path)
