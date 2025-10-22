import numpy as np
import networkx as nx
import random
import pandas as pd
import csv

# Paths to the CSV files
path_costs_csv = 'costs.csv'
friendships_csv = 'NoseBook_friendships.csv'

# Reading the CSV files
df_costs = pd.read_csv(path_costs_csv)
df_friendships = pd.read_csv(friendships_csv)

# Creating a directed graph G
G = nx.DiGraph()
for i in range(len(df_costs)):
    G.add_node(int(i))

# Adding edges to the graph
for tuple in df_friendships.itertuples():
    G.add_edge(int(tuple[1]), int(tuple[2]))

# Creating the inverse graph G_inv
G_inv = nx.DiGraph()
for i in range(len(df_costs)):
    G_inv.add_node(int(i))

# Adding reversed edges to G_inv
for tuple in df_friendships.itertuples():
    G_inv.add_edge(int(tuple[2]), int(tuple[1]))

# Calculating out-degrees
out_degrees = dict(G.out_degree())

# Calculate ancestors for each node
ancestors = {node: set(nx.ancestors(G, node)) for node in G.nodes()}

# Identify nodes to remove
nodes_to_remove = [
    node for node in G.nodes()
    if out_degrees[node] < 5 and len(ancestors[node]) < 3
]

# Remove nodes from the DAG
G.remove_nodes_from(nodes_to_remove)

# Paths to the files
NoseBook_path = friendships_csv
cost_path = path_costs_csv

# Function to submit the influencers
def influencers_submission(ID1, ID2, lst):
    with open(f'{ID1}_{ID2}.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(lst)

# Function to create the NoseBook graph from friendship relations
def create_graph(edges_path: str) -> nx.Graph:
    edges = pd.read_csv(edges_path).to_numpy()
    net = nx.Graph()
    net.add_edges_from(edges)
    return net

# Function to simulate a purchase round
def buy_products(net: nx.Graph, purchased: set) -> set:
    new_purchases = set()
    for user in net.nodes:
        neighborhood = set(net.neighbors(user))
        b = len(neighborhood.intersection(purchased))
        n = len(neighborhood)
        prob = b / n
        if prob >= random.uniform(0, 1):
            new_purchases.add(user)
    return new_purchases.union(purchased)

# Function to calculate the product exposure score
def product_exposure_score(net: nx.Graph, purchased_set: set) -> int:
    exposure = 0
    for user in net.nodes:
        neighborhood = set(net.neighbors(user))
        if user in purchased_set:
            exposure += 1
        elif len(neighborhood.intersection(purchased_set)) != 0:
            b = len(neighborhood.intersection(purchased_set))
            rand = random.uniform(0, 1)
            if rand < 1 / (1 + 10 * np.exp(-b/2)):
                exposure += 1
    return exposure

# Function to get the cost of chosen influencers
def get_influencers_cost(cost_path: str, influencers: list) -> int:
    costs = pd.read_csv(cost_path)
    return sum([costs[costs['user'] == influencer]['cost'].item() if influencer in list(costs['user']) else 0 for influencer in influencers])

# Function to launch the simulation with chosen influencers
def launch(influencers):
    NoseBook_network = create_graph(NoseBook_path)
    influencers_cost = get_influencers_cost(cost_path, influencers)
    if influencers_cost > 1000:
        exit()
    purchased = set(influencers)
    for i in range(6):
        purchased = buy_products(NoseBook_network, purchased)
    score = product_exposure_score(NoseBook_network, purchased)
    return score

# Calculate adjusted ancestors for each node
dict_ancestors = {}
dict_ancestors_count = {}
nodes = G.nodes()
for node in nodes:
    dict_ancestors[node] = list(nx.ancestors(G, node))
for node in nodes:
    for element in dict_ancestors[node]:
        if nx.shortest_path_length(G_inv, source=node, target=element) > 6:
            dict_ancestors[node].remove(element)
    dict_ancestors_count[node] = len(dict_ancestors[node])

# Calculate betweenness centrality
dict_betweeness = nx.betweenness_centrality(G)

# Calculate eigenvector centrality
dict_eigenvectors_centrality = nx.eigenvector_centrality(G, max_iter=1000)

# Function to normalize and adjust by cost for each centrality measure
def normalize_and_adjust(centrality_dict, df_costs):
    values = list(centrality_dict.values())
    min_value = min(values)
    max_value = max(values)
    if min_value == max_value:
        normalized = {key: 1.0 for key in centrality_dict}
    else:
        normalized = {key: (value - min_value) / (max_value - min_value)
                      for key, value in centrality_dict.items()}
    adjusted = {key: normalized_value / df_costs.loc[key, 'cost']
                for key, normalized_value in normalized.items()}
    return adjusted

# Normalize and adjust by cost for each centrality measure
adjusted_ancestors = normalize_and_adjust(dict_ancestors_count, df_costs)
adjusted_betweeness = normalize_and_adjust(dict_betweeness, df_costs)
adjusted_eigenvectors = normalize_and_adjust(dict_eigenvectors_centrality, df_costs)

# Coefficients for centrality measures
ancestors_coeff = 0.25
betweeness_coeff = 0.25
eigenvectors_coeff = 0.5

# Combine adjusted centrality values
combined_centrality = {}
for node in G.nodes():
    combined_value = (
        adjusted_ancestors.get(node, 0) * ancestors_coeff +
        adjusted_betweeness.get(node, 0) * betweeness_coeff +
        adjusted_eigenvectors.get(node, 0) * eigenvectors_coeff
    )
    combined_centrality[node] = combined_value

# Create a list of (node, combined_value) pairs and sort
combined_centrality_pairs = list(combined_centrality.items())
sorted_keys = [key for key, value in sorted(combined_centrality_pairs, key=lambda item: item[1], reverse=True)]

# Number of top influencers to select
top_number = 25
top_x_nodes = sorted_keys[:top_number]

# Default number of simulations
num_simulations = 100

# Function to optimize influencers
def optimize_influencers(top_influencers, num_simulations=num_simulations):
    best_influencers = []
    best_expectation = 0
    nodes = top_influencers
    sample_trials = 5000  # Number of random trials
    filtered_costs = df_costs[df_costs['user'].isin(top_influencers)]
    min_cost = filtered_costs['cost'].min()

    # Perform a number of random trials to optimize influencer selection
    for i in range(sample_trials):
        initial_influencers = []
        total_cost = 0
        budget = 1000

        # Add influencers until the budget is exceeded
        while total_cost <= budget - min_cost:
            candidate = random.choice(nodes)
            candidate_cost = df_costs.at[candidate, 'cost']
            if total_cost + candidate_cost <= budget and candidate not in initial_influencers:
                initial_influencers.append(candidate)
                total_cost += candidate_cost

        count = 0
        for j in range(num_simulations):
            if j == 20:
                middle_mean = count / 20
                if middle_mean < 1100:
                    print('trial is broken')
                    break
            count += launch(initial_influencers)

        expectation = count / num_simulations

        print("Sample trial number " + str(i) + ", influencers are : ")
        print(initial_influencers)
        print("Expectation is : " + str(expectation))
        print()

        if expectation > best_expectation:
            best_expectation = expectation
            best_influencers = initial_influencers

        if i % 10 == 0:
            print("##############")
            print('Best until now is :')
            print(best_influencers)
            print('With expectation of : ' + str(best_expectation))
            print("##############")
            print()

    # Non-optimized influencers
    not_optimized_influencers = []
    for node in top_x_nodes:
        cost = df_costs.loc[df_costs['user'] == node, 'cost'].values[0]
        if budget - cost >= 0:
            budget -= cost
            not_optimized_influencers.append(node)
        if budget <= 0:
            break

    noi_count = 0
    for i in range(num_simulations):
        noi_count += launch(not_optimized_influencers)

    noi_score = noi_count / num_simulations

    if best_expectation < noi_score:
        print('Not optimized influencers is better')
        best_influencers = not_optimized_influencers
        best_expectation = noi_score
    else:
        print('Optimized influencers is better')

    print('Best influencers are : ' + str(best_influencers))
    print('The score is : ' + str(best_expectation))

    return best_influencers, best_expectation

# Call the function to optimize influencers
best_influencers, best_expectation = optimize_influencers(top_x_nodes)

# Submit the best influencers
influencers_submission(931215248, 345812259, best_influencers)
