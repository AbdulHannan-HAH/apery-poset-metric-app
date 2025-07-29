import matplotlib
matplotlib.use('Agg')  # Prevent tkinter GUI backend errors
from flask import Flask, request, jsonify, send_file, render_template
from io import BytesIO
import matplotlib.pyplot as plt
import networkx as nx
from itertools import combinations
import numpy as np

app = Flask(__name__, template_folder='templates')

def generate_semigroup(generators, max_check=3000):
    S = set()
    frontier = [0]
    while frontier:
        current = frontier.pop(0)
        if current > max_check:
            continue
        if current not in S:
            S.add(current)
            for g in generators:
                frontier.append(current + g)
    return S

def compute_apery_set(S, m):
    apery = {}
    for i in range(m):
        k = i
        while k <= max(S) + m:
            if k in S:
                apery[i] = k
                break
            k += m
    return sorted(apery.values())

def build_apery_graph(apery_set, S):
    G = nx.Graph()
    for a in apery_set:
        G.add_node(a)
    edge_list = []
    for i, j in combinations(apery_set, 2):
        if abs(i - j) in S:
            G.add_edge(i, j)
            edge_list.append((i, j, abs(i - j)))
    return G, edge_list

def is_resolving_set(G, landmarks):
    seen = {}
    for node in G.nodes:
        vector = tuple(
            nx.shortest_path_length(G, source=node, target=lm) if nx.has_path(G, node, lm) else -1
            for lm in landmarks
        )
        if vector in seen.values():
            return False
        seen[node] = vector
    return True

def find_metric_dimension(G):
    nodes = list(G.nodes)
    for r in range(1, len(nodes)):
        for landmark_set in combinations(nodes, r):
            if is_resolving_set(G, landmark_set):
                return r, landmark_set
    return len(nodes), nodes

def get_layout(G, nodes):
    n = len(nodes)
    if n <= 25:
        return nx.circular_layout(G)
    elif n <= 60:
        return nx.kamada_kawai_layout(G)
    else:
        return nx.spring_layout(G, seed=42)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.get_json()
    generators = sorted(set(data.get("generators", [])))

    try:
        if not generators or min(generators) <= 0:
            return jsonify({"error": "Please enter valid positive integers"})

        m = min(generators)
        S = generate_semigroup(generators)
        apery = compute_apery_set(S, m)
        G, edges = build_apery_graph(apery, S)
        md_value, md_set = find_metric_dimension(G)

        result = {
            "generators": generators,
            "min_generator": m,
            "apery_set": apery,
            "num_nodes": len(apery),
            "num_edges": len(edges),
            "metric_dimension": md_value,
            "resolving_set": sorted(md_set),
            "sample_edges": edges[:5],
        }

        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route("/plot", methods=["POST"])
def plot_graph():
    data = request.get_json()
    generators = sorted(set(data.get("generators", [])))

    try:
        if not generators or min(generators) <= 0:
            return jsonify({"error": "Please enter valid positive integers"})

        m = min(generators)
        S = generate_semigroup(generators)
        apery = compute_apery_set(S, m)
        G, _ = build_apery_graph(apery, S)
        pos = get_layout(G, apery)

        plt.figure(figsize=(8, 8))
        nx.draw(
            G, pos, with_labels=True,
            node_color='skyblue', edgecolors='black',
            node_size=300, font_size=8
        )
        plt.title(f"Apery Graph for <{', '.join(map(str, generators))}>")
        plt.axis("off")

        buf = BytesIO()
        plt.savefig(buf, format="png")
        plt.close()
        buf.seek(0)

        return send_file(buf, mimetype="image/png")
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route("/chatbot", methods=["POST"])
def chatbot():
    data = request.get_json()
    user_msg = data.get("message", "").lower()

    if any(word in user_msg for word in ["apery", "apéry"]):
        response = (
            "Apery Set (Apéry Set) properties:\n"
            "1. For semigroup S and m ∈ S, the Apery set is Ap(S,m) = {a₀,a₁,...,aₘ₋₁}\n"
            "2. Where aᵢ is the smallest element in S congruent to i modulo m\n"
            "3. Forms a complete residue system modulo m\n"
            "4. Key for studying numerical semigroups"
        )
    
    elif any(word in user_msg for word in ["metric", "dimension"]):
        response = (
            "Metric Dimension:\n"
            "1. Minimum number of landmarks needed to uniquely identify all nodes\n"
            "2. A set W is resolving if every node has unique distances to W\n"
            "3. Applications in network navigation and chemistry\n"
            "4. Computed by checking all possible landmark combinations"
        )
    
    elif any(word in user_msg for word in ["edge", "edges"]):
        response = (
            "Edges in Apery graphs connect two elements when:\n"
            "1. The absolute difference belongs to the semigroup\n"
            "2. Represent structural relationships in the semigroup\n"
            "3. Example: In <4,7>, |11-7|=4 ∈ <4,7> creates an edge\n"
            "4. The edge count affects the graph's connectivity"
        )
    
    elif any(word in user_msg for word in ["graph", "structure"]):
        response = (
            "Apery graph properties:\n"
            "1. Undirected simple graph\n"
            "2. Vertex set = Apery set elements\n"
            "3. Edge set = {(a,b) | |a-b| ∈ S}\n"
            "4. The graph's metric dimension reveals structural properties"
        )
    
    elif any(word in user_msg for word in ["semigroup", "numerical"]):
        response = (
            "Numerical semigroups:\n"
            "1. Additive submonoids of ℕ with finite complement\n"
            "2. Generated by coprime integers n₁,n₂,...,nₖ\n"
            "3. The Apery set encodes important structural information\n"
            "4. Applications in algebraic geometry and coding theory"
        )
    
    elif any(word in user_msg for word in ["generator", "input"]):
        response = (
            "Generator requirements:\n"
            "1. Must be positive integers\n"
            "2. Should be coprime for nontrivial structure\n"
            "3. Typical examples: <4,7>, <5,6,7>\n"
            "4. The minimal generator determines the Apery set size"
        )
    
    elif any(word in user_msg for word in ["How are you","Hi", "Whats going on"]):
        response = "I am fine and ready to answer your questions about Apery sets and metric dimension."
    
    else:
        response = (
            "I can explain these concepts:\n"
            "1. Apery sets\n"
            "2. Metric dimension\n"
            "3. Edge formation rules\n"
            "4. Graph structure\n"
            "5. Semigroup theory\n"
            "Ask about any of these topics!"
        )

    return jsonify({"response": response})

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)