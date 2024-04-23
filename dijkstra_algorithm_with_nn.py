import csv
import json
import matplotlib.pyplot as plt
import heapq
import torch
import torch.nn as nn

# Neural Network Model architecture
class EnhancedTravelTimePredictor(nn.Module):
    def __init__(self):
        super(EnhancedTravelTimePredictor, self).__init__()
        self.fc1 = nn.Linear(8, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.dropout1 = nn.Dropout(0.25)
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.dropout2 = nn.Dropout(0.25)
        self.fc3 = nn.Linear(64, 32)
        self.bn3 = nn.BatchNorm1d(32)
        self.fc4 = nn.Linear(32, 1)

    def forward(self, x):
        x = torch.relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        x = torch.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        x = torch.relu(self.bn3(self.fc3(x)))
        x = self.fc4(x)
        return x

'''
Function to load the adjacency matrix from a CSV file
args:
    file_path: str - path to the CSV file
returns:
    adjacency_matrix: list[list[str]] - adjacency matrix with edge details
'''
def load_adjacency_matrix_from_csv(file_path):
    nodes = set()
    with open(file_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            nodes.add(row['Node_Start'])
            nodes.add(row['Node_End'])

    node_to_index = {node: i for i, node in enumerate(sorted(nodes))}
    size = len(nodes)
    adjacency_matrix = [[None for _ in range(size)] for _ in range(size)]

    with open(file_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            start_idx = node_to_index[row['Node_Start']]
            end_idx = node_to_index[row['Node_End']]
            times = [float(row[f'Data{i}']) for i in range(1, 51)]
            average_time = sum(times) / len(times)
            distance = float(row['Length'])
            edge_details = {
                'Time': average_time,
                'Distance': distance,
                'Coordinates': {
                    'Start': [row['Latitude_Start'], row['Longitude_Start']],
                    'End': [row['Latitude_End'], row['Longitude_End']]
                }
            }
            adjacency_matrix[start_idx][end_idx] = json.dumps(edge_details)

    return adjacency_matrix, node_to_index

'''
Function to perform Dijkstra's algorithm on the adjacency matrix
args:
    adjacency_matrix: list[list[str]] - adjacency matrix with edge details
    node_mapping: dict{str: int} - mapping of node names to indices
    start_node: str - starting node for the algorithm
    model: EnhancedTravelTimePredictor - neural network model for travel time prediction
returns:
    min_times: list[float] - minimum travel times to each node
    path_distances: list[float] - total distances to each node
    previous_nodes: list[int] - previous node indices for each node
'''
def dijkstra_algorithm(adjacency_matrix, node_mapping, start_node, model=None):
    start_index = node_mapping[start_node]
    num_nodes = len(adjacency_matrix)
    min_times = [float('inf')] * num_nodes
    min_times[start_index] = 0
    path_distances = [0] * num_nodes
    priority_queue = [(0, start_index)]
    previous_nodes = [-1] * num_nodes

    while priority_queue:
        current_time, current_index = heapq.heappop(priority_queue)

        if current_time > min_times[current_index]:
            continue

        for neighbor_index, edge_data in enumerate(adjacency_matrix[current_index]):
            if edge_data is not None:
                edge_details = json.loads(edge_data)
                if model:
                    # Use the model for prediction
                    input_features = torch.tensor([[
                        0,  # Dummy 'Link' identifier if necessary
                        current_index,
                        float(edge_details['Coordinates']['Start'][1]),
                        float(edge_details['Coordinates']['Start'][0]),
                        neighbor_index,
                        float(edge_details['Coordinates']['End'][1]),
                        float(edge_details['Coordinates']['End'][0]),
                        edge_details['Distance']
                    ]], dtype=torch.float32)
                    model.eval()
                    with torch.no_grad():
                        travel_time = model(input_features).item()
                else:
                    # Use average time
                    travel_time = edge_details['Time']

                travel_distance = edge_details['Distance']
                total_time = current_time + travel_time

                if total_time < min_times[neighbor_index]:
                    min_times[neighbor_index] = total_time
                    path_distances[neighbor_index] = path_distances[current_index] + travel_distance
                    previous_nodes[neighbor_index] = current_index
                    heapq.heappush(priority_queue, (total_time, neighbor_index))

    return min_times, path_distances, previous_nodes

'''
Function to reconstruct the path from the previous nodes
args:
    previous_nodes: list[int] - previous node indices for each node
    node_mapping: dict{str: int} - mapping of node names to indices
    start_node: str - starting node
    end_node: str - ending node
returns:
    path: list[str] - reconstructed path
'''
def reconstruct_path(previous_nodes, node_mapping, start_node, end_node):
    if end_node not in node_mapping:
        return "End node not found in the graph."

    path = []
    current_node_index = node_mapping[end_node]
    while current_node_index != -1:
        current_node = list(node_mapping.keys())[list(node_mapping.values()).index(current_node_index)]
        path.append(current_node)
        current_node_index = previous_nodes[current_node_index]

    path.reverse()
    if path[0] == start_node:
        return path
    else:
        return "No path found."

'''
Function to plot the graph with the path
args:
    adjacency_matrix: list[list[str]] - adjacency matrix with edge details
    node_mapping: dict{str: int} - mapping of node names to indices
    path_raw: list[str] - path from raw Dijkstra's algorithm
    path_model: list[str] - path from model-based Dijkstra's algorithm
    title_raw: str - title for raw Dijkstra's plot
    title_model: str - title for model-based Dijkstra's plot
returns:
    None
'''
def plot_graph(adjacency_matrix, node_mapping, path_raw, path_model, title_raw, title_model):
    plt.figure(figsize=(24, 10))  # Adjust figure size for better visibility of both plots

    # Prepare for subplots: first subplot for raw Dijkstra's output
    plt.subplot(1, 2, 1)
    plot_single_graph(adjacency_matrix, node_mapping, path_raw, title_raw)

    # Second subplot for model-based Dijkstra's output
    plt.subplot(1, 2, 2)
    plot_single_graph(adjacency_matrix, node_mapping, path_model, title_model)

    plt.show()

'''
Function to plot a single graph with the path
args:
    adjacency_matrix: list[list[str]] - adjacency matrix with edge details
    node_mapping: dict{str: int} - mapping of node names to indices
    path: list[str] - path to highlight
    title: str - title for the plot
returns:
    None
'''
def plot_single_graph(adjacency_matrix, node_mapping, path, title):
    node_coords = {}
    for node, idx in node_mapping.items():
        for adj in adjacency_matrix[idx]:
            if adj is not None:
                edge_details = json.loads(adj)
                start_coords = [float(x) for x in edge_details['Coordinates']['Start']]
                end_coords = [float(x) for x in edge_details['Coordinates']['End']]
                node_coords[node] = start_coords

    # Draw all edges
    for i, row in enumerate(adjacency_matrix):
        start_node = list(node_mapping.keys())[list(node_mapping.values()).index(i)]
        start_coords = node_coords.get(start_node)
        if start_coords:
            for neighbor_index, adj in enumerate(row):
                if adj is not None:
                    edge_details = json.loads(adj)
                    end_coords = [float(x) for x in edge_details['Coordinates']['End']]
                    plt.plot([start_coords[1], end_coords[1]], [start_coords[0], end_coords[0]], color='blue', alpha=0.5, zorder=1)

    # Highlight path nodes and edges
    for node, coords in node_coords.items():
        node_color = 'red' if path and node in path else 'blue'
        plt.plot(coords[1], coords[0], 'o', color=node_color, markersize=8, zorder=3)  # plot nodes

    if path:
        for i in range(len(path) - 1):
            start_node = path[i]
            end_node = path[i + 1]
            start_coords = node_coords[start_node]
            end_coords = node_coords[end_node]
            plt.plot([start_coords[1], end_coords[1]], [start_coords[0], end_coords[0]], color='red', linewidth=2.5, zorder=2)  # plot path

    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title(title)
    plt.grid(True)

# Main Execution
if __name__ == "__main__":
    # Your setup code here
    file_path = './ChengDuData/Weekday_Peak.csv'
    adjacency_matrix, node_mapping = load_adjacency_matrix_from_csv(file_path)

    model_path = 'models/model4/model4.pth'
    model = EnhancedTravelTimePredictor()
    model.load_state_dict(torch.load(model_path))
    while True:
        start_node = input("Enter the start node <type exit to stop>: ")
        end_node = input("Enter the end node <type exit to stop>: ")
        if start_node == 'exit' or end_node == 'exit':
            break
        # Perform algorithms
        min_times_raw, path_distances_raw, previous_nodes_raw = dijkstra_algorithm(adjacency_matrix, node_mapping, start_node)
        path_raw = reconstruct_path(previous_nodes_raw, node_mapping, start_node, end_node)

        min_times_model, path_distances_model, previous_nodes_model = dijkstra_algorithm(adjacency_matrix, node_mapping, start_node, model)
        path_model = reconstruct_path(previous_nodes_model, node_mapping, start_node, end_node)

        # Output and plotting
        print("Raw Dijkstra - Minimum time:", min_times_raw[node_mapping[end_node]], "Total distance:", path_distances_raw[node_mapping[end_node]])
        print("Model-based Dijkstra - Minimum time:", min_times_model[node_mapping[end_node]], "Total distance:", path_distances_model[node_mapping[end_node]])
        plot_graph(adjacency_matrix, node_mapping, path_raw, path_model, f"Raw Dijkstra Algorithm Travel Time Prediction from {start_node} to {end_node}", f"Dijkstra Algorithm with Neural Network Travel Time Prediction from {start_node} to {end_node}")
