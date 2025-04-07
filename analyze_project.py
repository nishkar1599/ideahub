import pandas as pd
import networkx as nx

# Load data
df = pd.read_csv('idea_hub_project.csv')

# Build graph
def build_graph(df):
    G = nx.Graph()
    for _, row1 in df.iterrows():
        G.add_node(row1['Title'], domain=row1['Domain'])
        for _, row2 in df.iterrows():
            if row1['Title'] != row2['Title']:
                weight = 0
                # Same domain base connection
                if row1['Domain'] == row2['Domain']:
                    weight += 1

                # Connect based on technology similarities
                if 'Flask' in row1['Title'] and 'Flask' in row2['Title']:
                    weight += 1
                if 'Django' in row1['Title'] and 'Django' in row2['Title']:
                    weight += 1
                if 'React' in row1['Title'] and 'React' in row2['Title']:
                    weight += 1

                # Connect web projects with their likely related domains
                if row1['Domain'] == 'Web Development':
                    if row2['Domain'] in ['AI', 'ML'] and 'Dashboard' in row2['Title']:
                        weight += 1
                    if row2['Domain'] == 'NLP' and 'Chat' in row1['Title']:
                        weight += 1

                if weight > 0:
                    G.add_edge(row1['Title'], row2['Title'], weight=weight)
    return G

# Create graph
G = build_graph(df)

# Calculate centralities
eigenvector_centrality = nx.eigenvector_centrality(G)
degree_centrality = nx.degree_centrality(G)
betweenness_centrality = nx.betweenness_centrality(G)

# Analyze Web Development projects
print("\nAnalysis of Web Development Projects:")
print("-" * 50)

# Get all Web Development projects
web_projects = [p for p in G.nodes() if G.nodes[p]['domain'] == 'Web Development']

# Print all Web Development projects with their scores
print("\nAll Web Development Projects and Their Scores:")
for project in web_projects:
    print(f"\n{project}:")
    print(f"  Eigenvector Score: {eigenvector_centrality[project]:.4f}")
    print(f"  Degree Score: {degree_centrality[project]:.4f}")
    print(f"  Betweenness Score: {betweenness_centrality[project]:.4f}")

    # Print connections with weights
    print("  Connected to:")
    for neighbor in G.neighbors(project):
        weight = G[project][neighbor]['weight']
        print(f"    - {neighbor} (weight: {weight})")

# Print top 5 most influential Web Development projects
print("\nTop 5 Most Influential Web Development Projects:")
web_scores = {p: eigenvector_centrality[p] for p in web_projects}
sorted_web = sorted(web_scores.items(), key=lambda x: x[1], reverse=True)
for project, score in sorted_web[:5]:
    print(f"{project}: {score:.4f}")

# Print network statistics
print("\nWeb Development Network Statistics:")
print(f"Number of projects: {len(web_projects)}")
print(f"Number of connections: {G.subgraph(web_projects).number_of_edges()}")
print(f"Average connections per project: {2*G.subgraph(web_projects).number_of_edges()/len(web_projects):.2f}")
