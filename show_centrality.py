import pandas as pd
import networkx as nx

# Load data
df = pd.read_csv('idea_hub_project.csv')

# Build graph
def build_graph(df):
    G = nx.Graph()
    for _, row1 in df.iterrows():
        G.add_node(row1['Title'], domain=row1['Domain'], difficulty=row1['Difficulty'])
        for _, row2 in df.iterrows():
            if row1['Title'] != row2['Title']:
                if row1['Domain'] == row2['Domain']:
                    G.add_edge(row1['Title'], row2['Title'], weight=1)
    return G

# Create graph
G = build_graph(df)

# Calculate centralities
eigenvector_centrality = nx.eigenvector_centrality(G)

# Print results by domain
print("\nMost Influential Projects by Domain:")
domains = df['Domain'].unique()
for domain in domains:
    print(f"\n{domain} Projects:")
    domain_projects = df[df['Domain'] == domain]['Title'].tolist()
    domain_scores = {p: eigenvector_centrality[p] for p in domain_projects}
    sorted_scores = sorted(domain_scores.items(), key=lambda x: x[1], reverse=True)
    for project, score in sorted_scores:
        print(f"  {project}: {score:.4f}")
