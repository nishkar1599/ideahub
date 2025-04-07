from flask import Flask, render_template, request
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import os
import io
import base64

app = Flask(__name__)

# Load CSV and build graph
df = pd.read_csv('idea_hub_project.csv')

# Build Graph
def build_graph(df):
    G = nx.Graph()
    for _, row1 in df.iterrows():
        G.add_node(row1['Title'], domain=row1['Domain'], difficulty=row1['Difficulty'])
        for _, row2 in df.iterrows():
            if row1['Title'] != row2['Title']:
                weight = 0
                if row1['Domain'] == row2['Domain']:
                    weight += 2
                if row1['Difficulty'] == row2['Difficulty']:
                    weight += 1
                if weight > 0:
                    G.add_edge(row1['Title'], row2['Title'], weight=weight)
    return G

G = build_graph(df)
degree_centrality = nx.degree_centrality(G)
betweenness_centrality = nx.betweenness_centrality(G)
eigenvector_centrality = nx.eigenvector_centrality(G)

@app.route('/')
def index():
    df = pd.read_csv('idea_hub_project.csv')
    domains = sorted(df['Domain'].unique())
    difficulties = sorted(df['Difficulty'].unique())
    return render_template('index.html', domains=domains, difficulties=difficulties)

@app.route('/results', methods=['POST'])
def results():
    domain = request.form['domain']
    difficulty = request.form['difficulty']
    df = pd.read_csv('idea_hub_project.csv')
    filtered = df[(df['Domain'] == domain) & (df['Difficulty'] == difficulty)]
    return render_template('results.html', projects=filtered.to_dict(orient='records'))

@app.route('/sna-recommendation', methods=['GET', 'POST'])
def sna_recommendation():
    df['Title'] = df['Title'].str.strip()
    titles = df['Title'].tolist()
    deg_recommendations, btw_recommendations, eig_recommendations = [], [], []
    graph_image_path = None

    if request.method == 'POST':
        selected = request.form['project'].strip()

        def build_recommendations(centrality_dict):
            recommendations = [
                {
                    'title': n,
                    'weight': G[selected][n]['weight'],
                    'centrality': round(centrality_dict[n], 3),
                    'github': df[df['Title'] == n]['GitHub Link'].values[0],
                    'youtube': df[df['Title'] == n]['YouTube Link'].values[0]
                }
                for n in G.neighbors(selected) if n in centrality_dict
            ]
            # Sort by centrality descending
            recommendations = sorted(recommendations, key=lambda x: x['centrality'], reverse=True)
            return recommendations[:5]  # Limit to top 5

        if selected in G:
            deg_recommendations = build_recommendations(degree_centrality)
            btw_recommendations = build_recommendations(betweenness_centrality)
            eig_recommendations = build_recommendations(eigenvector_centrality)

            # Generate and save graph visualization
            plt.figure(figsize=(12, 8))
            pos = nx.spring_layout(G, seed=42)
            nx.draw(G, pos, with_labels=True, node_size=500, node_color="skyblue", font_size=8)
            nx.draw_networkx_nodes(G, pos, nodelist=[selected], node_color='orange')
            plt.title(f'SNA Graph - {selected}')
            graph_image_path = f'static/graph_{selected}.png'
            plt.savefig(graph_image_path, format='PNG')
            plt.close()

    return render_template(
        'sna_recommendation.html',
        titles=titles,
        deg_recommendations=deg_recommendations,
        btw_recommendations=btw_recommendations,
        eig_recommendations=eig_recommendations,
        graph_image=graph_image_path
    )

@app.route('/sna-graph')
def sna_graph():
    plt.figure(figsize=(14, 10))
    pos = nx.spring_layout(G, k=0.4)  # or nx.kamada_kawai_layout(G)
    nx.draw(G, pos, with_labels=True, node_size=900, font_size=8,
            node_color="#7c5fff", edge_color="#ddd", font_color='black')

    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight')
    img.seek(0)
    graph_url = base64.b64encode(img.getvalue()).decode()
    plt.close()
    return render_template('sna_graph.html', graph_url=graph_url)

if __name__ == '__main__':
    if not os.path.exists('static'):
        os.makedirs('static')
    app.run(debug=True)
