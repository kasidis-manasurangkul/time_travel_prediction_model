import csv
import json
import matplotlib.pyplot as plt
import json
import heapq
import torch
import torch.nn as nn

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
            average_time = sum(times) / len(times)  # Calculate the average time
            distance = float(row['Length'])  # Assume 'Length' is the distance
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



import matplotlib.pyplot as plt
import json
class EnhancedTravelTimePredictor(nn.Module):
    def __init__(self):
        super(EnhancedTravelTimePredictor, self).__init__()
        self.fc1 = nn.Linear(8, 128)  # Adjust input features if necessary
        self.bn1 = nn.BatchNorm1d(128)  # Corrected to match the output features of fc1
        self.dropout1 = nn.Dropout(0.25)
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.dropout2 = nn.Dropout(0.25)
        self.fc3 = nn.Linear(64, 32)
        self.bn3 = nn.BatchNorm1d(32)  # Corrected to match the output features of fc3
        self.fc4 = nn.Linear(32, 1)

    def forward(self, x):
        x = torch.relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        x = torch.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        x = torch.relu(self.bn3(self.fc3(x)))
        x = self.fc4(x)
        return x


def predict_travel_time(model_path, input_data):
    # Load the model
    model = EnhancedTravelTimePredictor()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    # Load the input data
    # Add an extra dimension to the input data to represent a batch of size 1
    input_data = torch.tensor(input_data, dtype=torch.float32).unsqueeze(0)
    
    # Make the prediction
    with torch.no_grad():
        prediction = model(input_data)
    
    # Return the predicted value as a Python float
    return prediction.item()


def plot_graph(adjacency_matrix, node_mapping, path=None):
    plt.figure(figsize=(12, 8))

    # Extract and plot each node's coordinates
    node_coords = {}
    for node, idx in node_mapping.items():
        for adj in adjacency_matrix[idx]:
            if adj is not None:
                edge_details = json.loads(adj)
                start_coords = [float(x) for x in edge_details['Coordinates']['Start']]
                end_coords = [float(x) for x in edge_details['Coordinates']['End']]
                node_coords[node] = start_coords

    # First draw all edges in blue to ensure the red path will be on top
    for i, row in enumerate(adjacency_matrix):
        start_node = list(node_mapping.keys())[list(node_mapping.values()).index(i)]
        start_coords = node_coords.get(start_node)
        if start_coords:
            for neighbor_index, adj in enumerate(row):
                if adj is not None:
                    edge_details = json.loads(adj)
                    end_coords = [float(x) for x in edge_details['Coordinates']['End']]
                    plt.plot([start_coords[1], end_coords[1]], [start_coords[0], end_coords[0]], color='blue', alpha=0.5, zorder=1)

    # Now plot nodes and the selected path with higher zorder and thicker lines
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
    plt.title('Graph Visualization')
    plt.grid(True)
    plt.show()


import heapq
import json

def dijkstra_algorithm(adjacency_matrix, node_mapping, start_node):
    if start_node not in node_mapping:
        return "Start node not found in the graph.", None

    start_index = node_mapping[start_node]
    num_nodes = len(adjacency_matrix)
    min_times = [float('inf')] * num_nodes
    min_times[start_index] = 0
    path_distances = [0] * num_nodes  # Track total distance for each node
    priority_queue = [(0, start_index)]
    previous_nodes = [-1] * num_nodes

    while priority_queue:
        current_time, current_index = heapq.heappop(priority_queue)

        if current_time > min_times[current_index]:
            continue

        for neighbor_index, edge_data in enumerate(adjacency_matrix[current_index]):
            if edge_data is not None:
                edge_details = json.loads(edge_data)
                travel_time = edge_details['Time']
                travel_distance = edge_details['Distance']
                total_time = current_time + travel_time

                if total_time < min_times[neighbor_index]:
                    min_times[neighbor_index] = total_time
                    path_distances[neighbor_index] = path_distances[current_index] + travel_distance
                    previous_nodes[neighbor_index] = current_index
                    heapq.heappush(priority_queue, (total_time, neighbor_index))

    return min_times, path_distances, previous_nodes




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



# Replace this with your actual file path
# file_path = './ChengDuData/Weekday_Peak.csv'
# adjacency_matrix, node_mapping = load_adjacency_matrix_from_csv(file_path)

# # Assuming adjacency_matrix and node_mapping have been loaded
# start_node = '0'  # Example starting node ID
# end_node = '148'   # Example ending node ID

# # Perform Dijkstra's algorithm
# min_times, path_distances, previous_nodes = dijkstra_algorithm(adjacency_matrix, node_mapping, start_node)

# # Reconstruct the shortest path from start_node to end_node
# path = reconstruct_path(previous_nodes, node_mapping, start_node, end_node)

# # Output results
# print("Minimum time from", start_node, "to", end_node, "is:", min_times[node_mapping[end_node]], "units")
# print("Total distance for this path is:", path_distances[node_mapping[end_node]], "units")
# print("Path:", path)

# # Call the function to plot the graph
# plot_graph(adjacency_matrix, node_mapping, path)
model_path = 'models/model3/model3.pth'
input_data = [3,1,104.0625393,30.7390774,311,104.0600238,30.74269338,467.5522935]
prediction = predict_travel_time(model_path, input_data)
print(f'Predicted travel time: {prediction:.4f} minutes')