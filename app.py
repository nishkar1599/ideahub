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

def get_main_technology(technologies):
    # Define priority of technologies
    tech_priority = {
        # AI/ML
        'TensorFlow': 5,
        'PyTorch': 5,
        'scikit-learn': 4,
        'GPT-3': 5,

        # Web
        'React': 4,
        'Django': 4,
        'Flask': 3,

        # NLP
        'BERT': 5,
        'NLTK': 4,
        'Transformers': 5,

        # CV
        'OpenCV': 4,
        'YOLO': 5,
        'MediaPipe': 4,

        # Security
        'RSA': 4,
        'AES': 4,
        'JWT': 3,
        'Cryptography': 4
    }

    # Get the main technology based on priority
    main_tech = None
    max_priority = -1

    techs = [t.strip() for t in technologies.split(',')]
    for tech in techs:
        if tech in tech_priority and tech_priority[tech] > max_priority:
            main_tech = tech
            max_priority = tech_priority[tech]

    return main_tech

def get_tech_weight(tech):
    # Very distinctive technologies (highest weight)
    distinctive_tech = {
        'TensorFlow': 3,      # Advanced ML
        'PyTorch': 3,         # Advanced ML
        'BERT': 3,            # Advanced NLP
        'GPT-3': 3,          # Advanced AI
        'YOLO': 3,           # Advanced CV
        'Transformers': 3,    # Advanced NLP
        'Neo4j': 3,          # Specialized DB
        'MediaPipe': 3,       # Specialized CV
    }

    # Meaningful framework/libraries (medium weight)
    significant_tech = {
        'Flask': 2,          # Web Framework
        'Django': 2,         # Web Framework
        'React': 2,          # Frontend Framework
        'OpenCV': 2,         # Computer Vision
        'NLTK': 2,           # NLP
        'spaCy': 2,          # NLP
        'Keras': 2,          # ML
        'NetworkX': 2,       # Graph Analysis
        'Socket.io': 2,      # Real-time Communication
        'Redis': 2,          # Advanced DB
        'MongoDB': 2,        # NoSQL DB
        'PostgreSQL': 2,     # Advanced SQL DB
    }

    # Common/basic technologies (minimal weight)
    basic_tech = {
        'Python': 0.2,
        'JavaScript': 0.2,
        'HTML': 0.2,
        'CSS': 0.2,
        'Bootstrap': 0.2,
        'jQuery': 0.2,
        'SQLite': 0.2,
        'NumPy': 0.2,
        'Pandas': 0.2,
        'Matplotlib': 0.2,
    }

    # Return appropriate weight
    if tech in distinctive_tech:
        return distinctive_tech[tech]
    elif tech in significant_tech:
        return significant_tech[tech]
    elif tech in basic_tech:
        return basic_tech[tech]
    return 1  # default weight for other technologies

# Build Graph
def build_graph(df):
    G = nx.Graph()
    for _, row1 in df.iterrows():
        # Add node with domain only
        G.add_node(row1['Title'], domain=row1['Domain'])

        for _, row2 in df.iterrows():
            if row1['Title'] != row2['Title']:
                weight = 0

                # Same domain gets higher weight
                if row1['Domain'] == row2['Domain']:
                    weight += 4  # Increased from 1 to 4 to prioritize domain relationships

                # Main technology connection (lower weight)
                tech1 = get_main_technology(str(row1['Technologies'])) if pd.notna(row1['Technologies']) else None
                tech2 = get_main_technology(str(row2['Technologies'])) if pd.notna(row2['Technologies']) else None
                if tech1 and tech2 and tech1 == tech2:
                    tech_weight = get_tech_weight(tech1)
                    weight += tech_weight  # Use base tech weight without doubling

                if weight > 0:
                    G.add_edge(row1['Title'], row2['Title'], weight=weight)
    return G

G = build_graph(df)
degree_centrality = nx.degree_centrality(G)
betweenness_centrality = nx.betweenness_centrality(G)
eigenvector_centrality = nx.eigenvector_centrality(G)

@app.route('/')
def index():
    # Filter out NaN values and get unique domains
    domains = sorted(df['Domain'].dropna().unique())
    # Clean domain names by removing anything after '#'
    domains = [d.split('#')[0].strip() for d in domains]
    # Remove duplicates and sort
    domains = sorted(list(set(domains)))
    difficulties = sorted(df['Difficulty'].dropna().unique())
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
            return recommendations[:10]  # Limit to top 5

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
    plt.clf()  # Clear any existing plots

    # Create graph with only project nodes
    pos = nx.spring_layout(G, k=0.4)

    # Filter to ensure we only have project nodes
    project_nodes = [node for node in G.nodes() if node in df['Title'].values]

    # Create subgraph with only project nodes
    project_graph = G.subgraph(project_nodes)

    # Draw only project nodes
    nx.draw(project_graph, pos,
            labels={node: node for node in project_nodes},
            node_size=900,
            font_size=8,
            node_color="#7c5fff",
            edge_color="#ddd",
            font_color='black')

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
