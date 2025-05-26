import sys
import dgl
import torch
import pickle
import random
import numpy as np
import torch.nn.functional as F 

from sklearn.metrics import f1_score
from evaluate.metrics import calculate_metrics
from torch_geometric.utils import add_self_loops, remove_self_loops, to_undirected

# Function to evaluate the model on a given graph and mask
def evaluate(g, features, labels, mask, model, save_res_dir='none', logits = None):
    model.eval()
    if logits is None:
        with torch.no_grad():
            logits = model(g, features)

    loss_fn = torch.nn.CrossEntropyLoss()
    logits = logits[mask] 
    labels = labels[mask]
    loss = loss_fn(logits, labels)

    _, indices = torch.max(logits, dim=1)
    correct = torch.sum(indices == labels)
    metrics = correct.item() * 1.0 / len(labels)

    # Save results if directory is specified
    if save_res_dir != 'none':
        _, indices_ = torch.max(logits, dim=1)
        pickle.dump([logits.cpu(), indices_.cpu(), labels.cpu()], open(save_res_dir, 'wb'))

    f1 = f1_score(labels.cpu().numpy(), indices.cpu().numpy(), average='macro')

    return metrics, f1, loss.item()


# Evaluate degree bias in the model's predictions
def evaluate_degree_bias(g, features, labels, mask, model, degrees, logits=None):
    model.eval()
    if logits is None:
        with torch.no_grad():
            logits = model(g, features)
    _, logits_ = torch.max(logits, dim=1)

    mask = mask.cpu().numpy() if isinstance(mask, torch.Tensor) else mask
    logits_ = logits_.cpu().numpy() if isinstance(logits_, torch.Tensor) else logits_
    labels = labels.cpu().numpy() if isinstance(labels, torch.Tensor) else labels
    degrees = degrees.cpu().numpy() if isinstance(degrees, torch.Tensor) else degrees

    # Test nodes
    test_indices = np.where(mask)[0]
    test_degrees = degrees[test_indices]
    
    # Sort test nodes by degree
    sorted_indices = np.argsort(test_degrees)
    sorted_test_indices = test_indices[sorted_indices]
    
    # Split into head (top 33%) and tail (bottom 33%)
    n_test = len(test_indices)
    split_point = n_test // 3  # Approximately 33%
    tail_nodes = sorted_test_indices[:split_point]
    head_nodes = sorted_test_indices[-split_point:]

    # Overall metrics
    overall_acc, overall_f1 = calculate_metrics(logits_, labels, test_indices)
    
    # Head and tail metrics
    head_acc, head_f1 = calculate_metrics(logits_, labels, head_nodes)
    tail_acc, tail_f1 = calculate_metrics(logits_, labels, tail_nodes)

    return {
        'overall': {'accuracy': overall_acc, 'f1': overall_f1},
        'head': {'accuracy': head_acc, 'f1': head_f1},
        'tail': {'accuracy': tail_acc, 'f1': tail_f1}
    }

# Function to calculate homophily ratio for each node in the graph
def calculate_homophily_ratio(graph, labels):
    # Convert labels to PyTorch tensor if it is a NumPy array
    if isinstance(labels, np.ndarray):
        labels = torch.tensor(labels, dtype=torch.long)

    # Move labels to the same device as the graph
    device = graph.device
    labels = labels.to(device)

    # Number of nodes
    num_nodes = graph.num_nodes()

    # Extract edges
    src, dst = graph.edges()
    src, dst = src.long(), dst.long()

    # Check for same-class neighbors
    same_class_mask = labels[src] == labels[dst]

    # Count same-class neighbors for each node
    same_class_counts = torch.zeros(num_nodes, dtype=torch.float32, device=labels.device).scatter_add_(
        0, src, same_class_mask.float()
    )

    # Count total neighbors for each node
    degrees = torch.zeros(num_nodes, dtype=torch.float32, device=labels.device).scatter_add_(
        0, src, torch.ones_like(src, dtype=torch.float32, device=labels.device)
    )

    # Calculate homophily ratios (avoid division by zero for isolated nodes)
    homophily_ratios = torch.zeros(num_nodes, dtype=torch.float32, device=labels.device)
    mask = degrees > 0
    homophily_ratios[mask] = same_class_counts[mask] / degrees[mask]

    return homophily_ratios

# Function to evaluate structural disparity in the model's predictions
def evaluate_structural_disparity(g, features, labels, mask, model, logits=None):
    model.eval()
    if logits is None:
        with torch.no_grad():
            logits = model(g, features)
    _, logits_ = torch.max(logits, dim=1)

    mask = mask.cpu().numpy() if isinstance(mask, torch.Tensor) else mask
    logits_ = logits_.cpu().numpy() if isinstance(logits_, torch.Tensor) else logits_
    labels = labels.cpu().numpy() if isinstance(labels, torch.Tensor) else labels

    homophily_ratios = calculate_homophily_ratio(graph=g, labels=labels)

    # Test nodes
    test_indices = np.where(mask)[0]
    test_homophily_ratios = homophily_ratios[test_indices]
    
    # Sort test nodes by homophily ratio
    sorted_indices = np.argsort(test_homophily_ratios.cpu().numpy())
    sorted_test_indices = test_indices[sorted_indices]
    
    # Split into homophilous (top 33%) and heterophilous (bottom 33%) nodes
    n_test = len(test_indices)
    split_point = n_test // 3  
    heterophilous_nodes = sorted_test_indices[:split_point]
    homophilous_nodes = sorted_test_indices[-split_point:]

    # Overall metrics
    overall_acc, overall_f1 = calculate_metrics(logits_, labels, test_indices)
    
    # homophilous and heterophilous metrics
    homophilous_acc, homophilous_f1 = calculate_metrics(logits_, labels, homophilous_nodes)
    heterophilous_acc, heterophilous_f1 = calculate_metrics(logits_, labels, heterophilous_nodes)

    return {
        'overall': {'accuracy': overall_acc, 'f1': overall_f1},
        'homophilous': {'accuracy': homophilous_acc, 'f1': homophilous_f1},
        'heterophilous': {'accuracy': heterophilous_acc, 'f1': heterophilous_f1}
    }

# Function to prepare edge index for a DGL graph
def precompute_edge_index(g, add_loops=True, undirected=True):
    # Extract edges from the DGL graph
    src, dst = g.edges()
    
    # Combine source and destination nodes into a PyTorch tensor
    edge_index = torch.stack([src, dst], dim=0)  # Shape: [2, num_edges]
    
    # Convert to undirected graph if needed
    if undirected:
        edge_index = to_undirected(edge_index, num_nodes=g.num_nodes())
    
    # Remove self-loops
    edge_index, _ = remove_self_loops(edge_index)
    
    # Add self-loops
    if add_loops:
        edge_index, _ = add_self_loops(edge_index, num_nodes=g.num_nodes())
    
    return edge_index

# Function to evaluate edge removal impact on model performance
def evaluate_edge_removal(g, features, labels, mask, model, repeats = 5, logits=None):
    model.eval()
    # Ensure features and labels are on the correct device
    device = next(model.parameters()).device
    features = features.to(device)
    labels = labels.to(device)
    mask = mask.to(device)
    test_g = g.remove_self_loop()

    mask_np = mask.cpu().numpy()
    labels_np = labels.cpu().numpy()

    # Get all edges
    u, v = test_g.edges(order='eid')
    eids = np.arange(test_g.number_of_edges())

    # Move to CPU before converting to numpy
    u = u.cpu().numpy()
    v = v.cpu().numpy()

    # Build undirected edge list and mapping to edge IDs
    undirected_edges = {}
    for i in range(len(u)):
        src = u[i]
        dst = v[i]
        edge = (min(src, dst), max(src, dst))
        if edge not in undirected_edges:
            undirected_edges[edge] = []
        undirected_edges[edge].append(i)  # Append edge ID

    undirected_edge_list = list(undirected_edges.keys())
    total_num_undirected_edges = len(undirected_edge_list)

    percentages = [100 - (100 // (repeats - 1)) * i for i in range(repeats)]
    results = {}

    for percentage in percentages:
        num_edges_to_keep = int(percentage / 100 * total_num_undirected_edges)
        if num_edges_to_keep == 0:
            # Create an empty graph with the same number of nodes
            sg = dgl.graph(([], []), num_nodes=test_g.number_of_nodes())
        else:
            # Randomly select undirected edges to keep
            selected_edges = random.sample(undirected_edge_list, num_edges_to_keep)
            # Get the corresponding edge IDs in both directions
            edge_ids_to_keep = []
            for edge in selected_edges:
                edge_ids = undirected_edges[edge]
                edge_ids_to_keep.extend(edge_ids)
            # Create a subgraph with the selected edges
            sg = dgl.edge_subgraph(test_g, edge_ids_to_keep, relabel_nodes = False, store_ids=True)

        # Move sg to the appropriate device
        sg = sg.to(device)
        # Align node features directly (no need for indexing as node IDs are preserved)
        sg_features = features

        # Move sg to the appropriate device
        sg = sg.to(device)
        sg_features = sg_features.to(device)
        sg = sg.add_self_loop()

        # Evaluate model on the subgraph
        model.eval()
        with torch.no_grad():
            logits = model(sg, sg_features)
            _, logits_ = torch.max(logits, dim=1)
            logits_ = logits_.cpu().numpy()
        acc, f1 = calculate_metrics(logits_, labels_np, mask_np)

        results[percentage] = {'accuracy': acc, 'f1': f1}

    return results


