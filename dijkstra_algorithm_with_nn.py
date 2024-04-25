import csv
import json
import matplotlib.pyplot as plt
import heapq
import torch
import torch.nn as nn
import random
import sys

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
    try:
        with open(file_path, newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                nodes.add(row['Node_Start'])
                nodes.add(row['Node_End'])

    except FileNotFoundError:
        print("File not found. Please provide a valid file path.")
        sys.exit(1)
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
    end_node: str - ending node for the algorithm
    model: EnhancedTravelTimePredictor - neural network model for travel time prediction
returns:
    min_time: float - minimum travel time to the end node
    path_distance: float - total distance to the end node
    path: list[str] - shortest path from start to end node
    total_edge_time: float - total edge time used along the shortest path
'''
def dijkstra_algorithm(adjacency_matrix, node_mapping, start_node, end_node, model=None):
    start_index = node_mapping[start_node]
    end_index = node_mapping[end_node]
    num_nodes = len(adjacency_matrix)
    min_times = [float('inf')] * num_nodes
    min_times[start_index] = 0
    path_distances = [0] * num_nodes
    priority_queue = [(0, start_index)]
    previous_nodes = [-1] * num_nodes
    total_edge_time = 0

    while priority_queue:
        current_time, current_index = heapq.heappop(priority_queue)
        if current_index == end_index:
            break  # Reached the destination node, exit the loop
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

    # Reconstruct the shortest path and calculate the total edge time used
    path = []
    current_node = end_index
    while current_node != -1:
        path.append(list(node_mapping.keys())[list(node_mapping.values()).index(current_node)])
        if previous_nodes[current_node] != -1:
            edge_data = adjacency_matrix[previous_nodes[current_node]][current_node]
            edge_details = json.loads(edge_data)
            total_edge_time += edge_details['Time']
        current_node = previous_nodes[current_node]
    path.reverse()

    return min_times[end_index], path_distances[end_index], path, total_edge_time

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


def test_random_pairs(adjacency_matrix, node_mapping, num_pairs, model):
    actual_times = []
    predicted_times = []

    for _ in range(num_pairs):
        start_node, end_node = random.sample(list(node_mapping.keys()), 2)
        
        min_time_raw, _, _, _ = dijkstra_algorithm(adjacency_matrix, node_mapping, start_node, end_node)
        min_time_model, _, _, _ = dijkstra_algorithm(adjacency_matrix, node_mapping, start_node, end_node, model)
        
        actual_times.append(min_time_raw)
        predicted_times.append(min_time_model)

    # Plot the graph comparing actual and predicted times
    plt.figure(figsize=(8, 6))
    plt.scatter(actual_times, predicted_times, color='blue', alpha=0.5)
    plt.plot([min(actual_times), max(actual_times)], [min(actual_times), max(actual_times)], color='red', linestyle='--')
    plt.xlabel('Actual Minimum Time')
    plt.ylabel('Model-Predicted Minimum Time')
    plt.title('Comparison of Actual and Model-Predicted Minimum Times')
    plt.grid(True)
    plt.show()

    # Calculate the average error
    errors = [abs(actual - predicted) for actual, predicted in zip(actual_times, predicted_times)]
    avg_error = sum(errors) / len(errors)
    print(f"Average Error: {avg_error:.2f}")
    print(f'Average actual time: {sum(actual_times) / len(actual_times):.2f}')


# Main Execution
if __name__ == "__main__":
    # ask for the file path in args
    if len(sys.argv) < 2:
        print("Please provide the file path as an argument.")
        sys.exit(1)
    file_path = sys.argv[1]
    adjacency_matrix, node_mapping = load_adjacency_matrix_from_csv(file_path)

    model_path = 'models/model4/model4.pth'
    model = EnhancedTravelTimePredictor()
    model.load_state_dict(torch.load(model_path))
    while True:
        start_node = input("Enter the start node <type exit to stop>: ")
        if start_node == 'exit':
            break
        end_node = input("Enter the end node <type exit to stop>: ")
        if end_node == 'exit':
            break
        if start_node not in node_mapping or end_node not in node_mapping:
            print("Invalid node names. Please try again.")
            continue
        # Perform algorithms
        min_time_raw, path_distance_raw, path_raw, total_edge_time_raw = dijkstra_algorithm(adjacency_matrix, node_mapping, start_node, end_node)
        min_time_model, path_distance_model, path_model, total_edge_time_model = dijkstra_algorithm(adjacency_matrix, node_mapping, start_node, end_node, model)

        # Output and plotting
        print("Raw Dijkstra - Minimum time:", min_time_raw, "Total distance:", path_distance_raw, "Total edge time used:", total_edge_time_raw)
        print("Model-based Dijkstra - Minimum time:", min_time_model, "Total distance:", path_distance_model, "Total edge time used:", total_edge_time_model)
        plot_graph(adjacency_matrix, node_mapping, path_raw, path_model, f"Raw Dijkstra Algorithm Travel Time Prediction from {start_node} to {end_node}", f"Dijkstra Algorithm with Neural Network Travel Time Prediction from {start_node} to {end_node}")
    # test_random_pairs(adjacency_matrix, node_mapping, 500, model)