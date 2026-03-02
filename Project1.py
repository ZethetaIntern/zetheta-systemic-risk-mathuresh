import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random

# -------------------------------
# 1. CREATE FINANCIAL NETWORK
# -------------------------------

def create_financial_network(num_banks=10):
    G = nx.DiGraph()

    for i in range(num_banks):
        assets = random.randint(100, 1000)
        capital = assets * random.uniform(0.08, 0.15)
        G.add_node(
            f"Bank_{i}",
            assets=assets,
            capital=capital,
            failed=False
        )

    # Create exposures (edges)
    for i in range(num_banks):
        for j in range(num_banks):
            if i != j and random.random() < 0.3:
                exposure = random.randint(10, 200)
                G.add_edge(f"Bank_{i}", f"Bank_{j}", weight=exposure)

    return G


# -------------------------------
# 2. CALCULATE RISK METRICS
# -------------------------------

def calculate_risk_metrics(G):
    degree = nx.degree_centrality(G)
    betweenness = nx.betweenness_centrality(G)
    eigenvector = nx.eigenvector_centrality(G, max_iter=1000)

    systemic_scores = {}

    for node in G.nodes():
        score = (
            0.4 * degree[node] +
            0.3 * betweenness[node] +
            0.3 * eigenvector[node]
        )
        systemic_scores[node] = score
        G.nodes[node]['systemic_score'] = score

    return systemic_scores


# -------------------------------
# 3. CONTAGION SIMULATION
# -------------------------------

def contagion_simulation(G, shocked_bank):
    print(f"\n🔥 Initial Shock to: {shocked_bank}")

    G.nodes[shocked_bank]['failed'] = True
    failed_banks = {shocked_bank}

    while True:
        new_failures = set()

        for bank in failed_banks:
            for neighbor in G.successors(bank):
                if not G.nodes[neighbor]['failed']:
                    exposure = G[bank][neighbor]['weight']
                    G.nodes[neighbor]['capital'] -= exposure

                    print(f"💸 {neighbor} loses {exposure}, remaining capital: {G.nodes[neighbor]['capital']}")

                    if G.nodes[neighbor]['capital'] <= 0:
                        new_failures.add(neighbor)

        if not new_failures:
            break

        for bank in new_failures:
            G.nodes[bank]['failed'] = True

        failed_banks.update(new_failures)

    return failed_banks


# -------------------------------
# 4. STRESS TEST FUNCTION
# -------------------------------

def stress_test(G, shock_percent=0.3):
    print("\n📉 Running Market Shock Stress Test")

    for node in G.nodes():
        loss = G.nodes[node]['assets'] * shock_percent
        G.nodes[node]['capital'] -= loss

        if G.nodes[node]['capital'] <= 0:
            G.nodes[node]['failed'] = True

    failed = [n for n in G.nodes() if G.nodes[n]['failed']]
    return failed


# -------------------------------
# 5. VISUALIZATION
# -------------------------------

def visualize_network(G, title="Financial Network"):
    pos = nx.spring_layout(G)

    colors = []
    for node in G.nodes():
        if G.nodes[node]['failed']:
            colors.append("red")
        else:
            colors.append("green")

    weights = [G[u][v]['weight']/50 for u, v in G.edges()]

    plt.figure(figsize=(10, 7))
    nx.draw(
        G,
        pos,
        with_labels=True,
        node_color=colors,
        node_size=800,
        width=weights,
        font_size=8
    )
    plt.title(title)
    plt.show()


# -------------------------------
# 6. MAIN EXECUTION
# -------------------------------

if __name__ == "__main__":

    # Step 1: Create Network
    G = create_financial_network(10)

    # Step 2: Calculate Metrics
    scores = calculate_risk_metrics(G)

    print("\n📊 Systemic Importance Scores:")
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    for bank, score in sorted_scores:
        print(f"{bank}: {round(score, 4)}")

    # Step 3: Visualize Initial Network
    visualize_network(G, "Initial Financial Network")

    # Step 4: Shock Most Systemic Bank
    most_systemic = sorted_scores[0][0]
    failed_banks = contagion_simulation(G, most_systemic)

    print("\n❌ Failed Banks After Contagion:")
    print(failed_banks)

    # Step 5: Visualize After Contagion
    visualize_network(G, "After Contagion Simulation")