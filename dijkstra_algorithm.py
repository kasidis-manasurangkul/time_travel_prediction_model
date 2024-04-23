import csv
import json

def load_adjacency_matrix_from_csv(file_path):
    # First, collect all unique nodes to determine the size of the matrix
    nodes = set()
    with open(file_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            nodes.add(row['Node_Start'])
            nodes.add(row['Node_End'])

    # Create a mapping of nodes to indices
    node_to_index = {node: i for i, node in enumerate(nodes)}

    # Initialize the adjacency matrix
    size = len(nodes)
    adjacency_matrix = [[None for _ in range(size)] for _ in range(size)]

    # Populate the matrix
    with open(file_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            start_idx = node_to_index[row['Node_Start']]
            end_idx = node_to_index[row['Node_End']]
            edge_details = {
                'Length': row['Length'],
                'Coordinates': {
                    'Start': [row['Latitude_Start'], row['Longitude_Start']],
                    'End': [row['Latitude_End'], row['Longitude_End']]
                },
                'Data': {f'Data{i}': row[f'Data{i}'] for i in range(1, 51)}
            }
            adjacency_matrix[start_idx][end_idx] = json.dumps(edge_details)

    return adjacency_matrix, node_to_index

# Replace this with your actual file path
file_path = './ChengDuData/Weekday_Peak.csv'
adjacency_matrix, node_mapping = load_adjacency_matrix_from_csv(file_path)

# For demonstration: print part of the matrix
for row in adjacency_matrix[:50]:  # Adjust the slice as needed
    print(row[:50])  # Adjust the slice as needed

import matplotlib.pyplot as plt
import json

def plot_graph(adjacency_matrix, node_mapping):
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
                plt.plot(start_coords[1], start_coords[0], 'bo')  # plot node
                plt.plot(end_coords[1], end_coords[0], 'bo')  # plot node

    # Draw lines for each connection
    for i, row in enumerate(adjacency_matrix):
        start_node = list(node_mapping.keys())[list(node_mapping.values()).index(i)]
        start_coords = node_coords.get(start_node)

        if start_coords:
            for adj in row:
                if adj is not None:
                    edge_details = json.loads(adj)
                    end_coords = [float(x) for x in edge_details['Coordinates']['End']]
                    plt.plot([start_coords[1], end_coords[1]], [start_coords[0], end_coords[0]], 'b-')  # plot edge

    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Graph Visualization')
    plt.grid(True)
    plt.show()

# Call the function to plot the graph
plot_graph(adjacency_matrix, node_mapping)
