import torch
import dgl
import os
import pickle 
import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold

def print_graph_statistics(g, data_name=None):
    num_nodes = g.num_nodes()
    num_edges = g.num_edges()
    num_feats = g.ndata['feat'].shape[1] if 'feat' in g.ndata else 0
    num_classes = len(torch.unique(g.ndata['label']))
    num_train_samples = g.ndata['train_mask'].sum().item()
    num_val_samples = g.ndata['val_mask'].sum().item()
    num_test_samples = g.ndata['test_mask'].sum().item()

    # Calculate the homophily ratio (count each bidirectional edge only once)
    edge_src, edge_dst = g.edges()
    labels = g.ndata['label']

    # Ensure each bidirectional edge is counted once by filtering u < v
    valid_edges = edge_src < edge_dst
    edge_src = edge_src[valid_edges]
    edge_dst = edge_dst[valid_edges]

    # Calculate homophily: edges where labels are the same
    same_label_edges = (labels[edge_src] == labels[edge_dst]).sum().item()
    homophily_ratio = same_label_edges / valid_edges.sum().item()

    print("-" * 50)
    print(f"Dataset: {data_name if data_name else 'Unknown'}")
    print(f"NumNodes: {num_nodes}")
    print(f"NumEdges: {num_edges}")
    print(f"NumFeats: {num_feats}")
    print(f"NumClasses: {num_classes}")
    print(f"NumTrainingSamples: {num_train_samples}")
    print(f"NumValidationSamples: {num_val_samples}")
    print(f"NumTestSamples: {num_test_samples}")
    print(f"HomophilyRatio: {homophily_ratio:.4f}")

def repeated_random_split(y, k_fold=5, n_iters=10):
    train_splits = []
    val_splits = []
    test_splits = []
    skf = RepeatedStratifiedKFold(n_splits=k_fold, n_repeats=n_iters, random_state=42)
    
    for i, (larger_group, smaller_group) in enumerate(skf.split(y, y)):
        train_y = y[smaller_group]
        sub_skf = StratifiedKFold(n_splits=2)
        train_split, val_split = next(iter(sub_skf.split(train_y, train_y)))
        train = torch.zeros_like(y, dtype=torch.bool)
        train[smaller_group[train_split]] = True
        val = torch.zeros_like(y, dtype=torch.bool)
        val[smaller_group[val_split]] = True
        test = torch.zeros_like(y, dtype=torch.bool)
        test[larger_group] = True
        train_splits.append(train.unsqueeze(1))
        val_splits.append(val.unsqueeze(1))
        test_splits.append(test.unsqueeze(1))
    
    return torch.cat(train_splits, dim=1), torch.cat(val_splits, dim=1), torch.cat(test_splits, dim=1)

def handle_splits(g, data_name, index=0):
    split_path = f'splits/{data_name}.splits'
    if os.path.exists(split_path):
        print(f"Loaded splits for {data_name} from {split_path}")
        all_splits = pickle.load(open(split_path, 'rb'))
    else:
        all_splits = repeated_random_split(g.ndata['label'])
        print(f"Generated splits for {data_name}: {all_splits[0].shape}")
        pickle.dump(all_splits, open(split_path, 'wb'))

    # Access the correct split using the index
    train_masks, val_masks, test_masks = all_splits
    g.ndata['train_mask'] = train_masks[:, index]
    g.ndata['val_mask'] = val_masks[:, index]
    g.ndata['test_mask'] = test_masks[:, index]

    return g

def load_dataset(data_name, index=0):
    os.makedirs('splits', exist_ok=True)

    random_split_dataset_map = {
        'cora': dgl.data.CoraGraphDataset,
        'citeseer': dgl.data.CiteseerGraphDataset,
        'pubmed': dgl.data.PubmedGraphDataset,
        'wiki_cs': dgl.data.WikiCSDataset,
        'co_photo': dgl.data.AmazonCoBuyPhotoDataset,
        'co_computer': dgl.data.AmazonCoBuyComputerDataset,
        'co_cs': dgl.data.CoauthorCSDataset,
        'co_phy': dgl.data.CoauthorPhysicsDataset
    }

    if data_name == 'actor':
        dataset = dgl.data.ActorDataset()
        g = dataset[0]
        if data_name == "actor":
            index = index % 5
        g.ndata['train_mask'] = g.ndata['train_mask'][:, index]
        g.ndata['val_mask'] = g.ndata['val_mask'][:, index]
        g.ndata['test_mask'] = g.ndata['test_mask'][:, index]
    elif data_name == 'chameleon' or data_name == 'squirrel':
        dataset = np.load(f'dataset/{data_name}_filtered.npz')
        node_features = torch.tensor(dataset['node_features'])
        edges = torch.tensor(dataset['edges'])
        g = dgl.graph((edges[:, 0], edges[:, 1]), num_nodes=node_features.shape[0], idtype=torch.int)
        g.ndata['feat'] = node_features
        g.ndata['label'] = torch.tensor(dataset['node_labels'])
        g.ndata['train_mask'] = torch.tensor(dataset['train_masks'][index])
        g.ndata['val_mask'] = torch.tensor(dataset['val_masks'][index])
        g.ndata['test_mask'] = torch.tensor(dataset['test_masks'][index])
    elif data_name == 'arxiv':
        from ogb.nodeproppred import DglNodePropPredDataset
        dataset = DglNodePropPredDataset(name='ogbn-arxiv')
        g, labels = dataset[0]
        g.ndata['label'] = labels.squeeze()
        splits = dataset.get_idx_split()
        g.ndata['train_mask'], g.ndata['val_mask'], g.ndata['test_mask'] = (
            torch.zeros(g.num_nodes(), dtype=torch.bool),
            torch.zeros(g.num_nodes(), dtype=torch.bool),
            torch.zeros(g.num_nodes(), dtype=torch.bool)
        )
        g.ndata['train_mask'][splits['train']] = True
        g.ndata['val_mask'][splits['valid']] = True
        g.ndata['test_mask'][splits['test']] = True
        # normalize graphs with discrete features
        from sklearn.preprocessing import StandardScaler
        norm = StandardScaler()
        norm.fit(g.ndata['feat'])
        g.ndata['feat'] = torch.tensor(norm.transform(g.ndata['feat'])).float()
    elif data_name in random_split_dataset_map:
        dataset = random_split_dataset_map[data_name]()
        g = dataset[0]
        handle_splits(g, data_name, index=index)
    else:
        raise ValueError('Invalid Dataset')

    g = g.remove_self_loop().add_self_loop()
    g = dgl.to_bidirected(g, copy_ndata=True)

    #print_graph_statistics(g, data_name)

    return g