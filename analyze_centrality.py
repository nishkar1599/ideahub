import pandas as pd
import networkx as nx
from collections import Counter

# Load data
df = pd.read_csv('idea_hub_project.csv')

def get_technologies(tech_str):
    if pd.isna(tech_str):
        return []
    return [t.strip() for t in tech_str.split(',')]

# Build graph with technology-based connections
def build_graph(df):
    G = nx.Graph()
    for _, row1 in df.iterrows():
        G.add_node(row1['Title'], domain=row1['Domain'])
        tech1 = get_technologies(row1['Technologies'])

        for _, row2 in df.iterrows():
            if row1['Title'] != row2['Title']:
                weight = 0
                # Same domain connection
                if row1['Domain'] == row2['Domain']:
                    weight += 2

                # Shared technologies
                tech2 = get_technologies(row2['Technologies'])
                shared_tech = set(tech1) & set(tech2)
                weight += len(shared_tech)

                if weight > 0:
                    G.add_edge(row1['Title'], row2['Title'], weight=weight)
    return G

# Create graph
G = build_graph(df)

# Calculate centralities
eigenvector_centrality = nx.eigenvector_centrality(G, weight='weight')
degree_centrality = nx.degree_centrality(G)
betweenness_centrality = nx.betweenness_centrality(G, weight='weight')

# Print top 10 projects by each centrality measure
print("\nTop 10 Most Central Projects (Eigenvector Centrality):")
print("-" * 70)
sorted_eigen = sorted(eigenvector_centrality.items(), key=lambda x: x[1], reverse=True)[:10]
for project, score in sorted_eigen:
    domain = G.nodes[project]['domain']
    print(f"{project} ({domain}): {score:.4f}")

print("\nTop 10 Most Connected Projects (Degree Centrality):")
print("-" * 70)
sorted_degree = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:10]
for project, score in sorted_degree:
    domain = G.nodes[project]['domain']
    print(f"{project} ({domain}): {score:.4f}")

print("\nTop 10 Bridge Projects (Betweenness Centrality):")
print("-" * 70)
sorted_between = sorted(betweenness_centrality.items(), key=lambda x: x[1], reverse=True)[:10]
for project, score in sorted_between:
    domain = G.nodes[project]['domain']
    print(f"{project} ({domain}): {score:.4f}")

# Analyze domain connections
print("\nDomain Connection Analysis:")
print("-" * 70)
domain_connections = {}
for edge in G.edges(data=True):
    domain1 = G.nodes[edge[0]]['domain']
    domain2 = G.nodes[edge[1]]['domain']
    weight = edge[2]['weight']
    if domain1 != domain2:
        key = tuple(sorted([domain1, domain2]))
        domain_connections[key] = domain_connections.get(key, 0) + weight

print("Strongest Cross-Domain Connections:")
sorted_connections = sorted(domain_connections.items(), key=lambda x: x[1], reverse=True)[:5]
for domains, weight in sorted_connections:
    print(f"{domains[0]} <-> {domains[1]}: {weight}")
