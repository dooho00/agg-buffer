import os
import dgl
import timeit
import torch
import torch.nn as nn
import torch.nn.functional as F

from copy import deepcopy
from data.load_dataset import load_dataset
from models.gnn import GCN, MLP, SAGE, GAT, SGC, GIN
from evaluate.utils import evaluate
from utils.utils import set_seed, save_model
from utils.log_utils import init_logger
from utils.arg_parser import TrainGNN

def train(g, features, labels, masks, model, args):
    # define masks and loss function
    train_mask = masks[0]
    val_mask = masks[1]
    test_mask = masks[2]
    loss_fcn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_eval, counter, patience, saved_model = 0, 0, 100, None

    # training loop
    for epoch in range(10000):
        if counter == patience:
            logger.info('Early Stopping...')
            break
        else:
            counter += 1

        model.train()
        logits = model(g, features) 
        loss = loss_fcn(logits[train_mask], labels[train_mask].long())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        acc, _, _ = evaluate(g, features, labels, val_mask, model)
        if acc > best_eval:
            best_eval = acc
            counter = 0
            saved_model = deepcopy(model)
            logger.info(
                "Epoch {:05d} | Loss {:.4f} | val acc {:.4f}".format(
                    epoch, loss.item(), acc
                )
            )
    
    # evaluate the final model on the test set
    model.load_state_dict(saved_model.state_dict())
    test_acc, _, _ = evaluate(g, features, labels, test_mask, model, '{}/{}.pkl'.format(args.save_dir, args.dataset))
    
    return test_acc, saved_model


if __name__ == "__main__":
    args = TrainGNN().parse()
    if not os.path.isdir(args.save_dir): os.makedirs(args.save_dir)
    logger = init_logger("main_logger", '{}/{}_run.log'.format(args.save_dir, args.dataset))

    # set the number of threads
    torch.set_num_threads(8)
    
    # load data
    print("-" * 50)
    g = load_dataset(args.dataset, index=args.index)
    device = torch.device("cuda:{}".format(args.device))
    g = g.int().to(device)
    features = g.ndata["feat"]
    labels = g.ndata["label"]
    masks = g.ndata["train_mask"], g.ndata["val_mask"], g.ndata["test_mask"]
    
    # random seed setting
    set_seed(args.seed)

    # model initialization
    in_size = features.shape[1]
    out_size = int(labels.max() + 1)

    model_args = {
        "in_feats": in_size,
        "hidden_lst": args.hid_dim + [out_size],
        "use_linear": args.use_linear,
        "dropout": args.dropout,
        "norm": args.norm,
    }

    # Special cases for models with different signatures
    model_factory = {
        "MLP": lambda: MLP(**model_args),
        "SAGE": lambda: SAGE(**model_args),
        "GIN": lambda: GIN(**model_args),
        "SGC": lambda: SGC(in_size, args.hid_dim + [out_size], use_linear=args.use_linear),
        "GAT": lambda: GAT(in_size, args.hid_dim[0], out_size, use_linear=args.use_linear, dropout=args.dropout, norm=args.norm),
        "GCN": lambda: GCN(**model_args, mp_norm=args.mp_norm),
    }

    # Fallback/default to GCN
    model_cls = model_factory.get(args.architecture, model_factory["GCN"])
    model = model_cls().to(device)
    print("-" * 50)
    print(model)
    print("-" * 50)

    # training the model
    logger.info("Training...")
    time = timeit.default_timer()
    test_acc, saved_model = train(g, features, labels, masks, model, args)
    logger.info("Test accuracy {:.2f}| Time Consumed: {:.2f} s".format(test_acc*100, timeit.default_timer()-time))
    save_model(saved_model.state_dict(), 
               {'in_feats':in_size, 
                'hidden_lst':args.hid_dim + [out_size],
                'norm':args.norm,
                'mp_norm':args.mp_norm},
               '{}/{}'.format(args.save_dir, args.dataset))
    print("Training complete. Model saved to {}/{}.ckpt".format(args.save_dir, args.dataset))
    print("-" * 50)