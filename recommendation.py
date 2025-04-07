import pandas as pd
import networkx as nx

csv_path = "idea_hub_project.csv"
def build_graph_from_csv(csv_path):
    df = pd.read_csv(csv_path)
    G = nx.Graph()

    for _, row in df.iterrows():
        project = row['Title']
        domain = row['Domain']
        difficulty = row['Difficulty']

        G.add_node(project, type='project', domain=domain, difficulty=difficulty)
        G.add_node(domain, type='domain')
        G.add_node(difficulty, type='difficulty')

        G.add_edge(project, domain)
        G.add_edge(project, difficulty)

    return G, df

def recommend_projects(G, input_project, top_n=5):
    if input_project not in G:
        return []

    neighbors = list(G.neighbors(input_project))
    scores = {}

    for neighbor in neighbors:
        for project in G.neighbors(neighbor):
            if project != input_project and G.nodes[project]['type'] == 'project':
                scores[project] = scores.get(project, 0) + 1

    recommended = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
    return [project for project, score in recommended]
