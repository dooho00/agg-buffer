import os
import dgl
import wandb
import timeit
import torch
import torch.nn as nn
import torch.nn.functional as F

from dgl import DropEdge
from copy import deepcopy
from data.load_dataset import load_dataset
from models.gnn import GCN, MLP, SAGE, GAT, SGC, GIN
from models.gnn_buffer import GCN_B, SAGE_B, GAT_B, GIN_B, SGC_B
from evaluate.utils import evaluate
from utils.utils import set_seed, save_model
from utils.log_utils import init_logger, evaluate_and_log_models
from utils.arg_parser import TrainBuffer

def train(g, features, masks, model, trained_model, args):
    # define masks
    train_mask = masks[0]
    val_mask = masks[1]
    test_mask = masks[2]

    # separate parameters based on their names
    buffer_params = []
    other_params = []
    for name, param in model.named_parameters():
        if 'buffer' in name:
            buffer_params.append(param)
        else:
            other_params.append(param)

    # define optimizer with different learning rates for parameter groups
    # freeze all layers except buffer
    optimizer = torch.optim.Adam([
        {'params': buffer_params, 'lr': args.lr, 'weight_decay': args.weight_decay},
        {'params': other_params, 'lr': 0.0, 'weight_decay': 0.0},
    ])
    for name, param in model.named_parameters():
        if not 'buffer' in name:
            param.requires_grad = False
        # print(f"Layer: {name} | Trainable: {param.requires_grad}")

    # get the original representation from the trained model
    trained_model.eval()
    with torch.no_grad():
        original_representation = trained_model(g, features, return_all = False)

    best_val_loss, counter, patience, saved_model = float('inf'), 0, 100, None
    transform_DropEdge = DropEdge(args.dropedge)
    eps = 1e-12

    # training loop
    for epoch in range(10000):
        model.train()
        optimizer.zero_grad()

        # get the representation of the original graph
        rep_original_g = model(g, features, return_all=False)

        # get the representation of the perturbed graph
        perturbed_g = g.remove_self_loop()
        perturbed_g = transform_DropEdge(g)
        perturbed_g = perturbed_g.add_self_loop()
        rep_perturbed_g = model(perturbed_g, features, return_all=False)

        # compute bias loss
        input_prob = F.log_softmax(rep_original_g[train_mask], dim=1) 
        target_prob = F.softmax(original_representation[train_mask], dim=1) 
        target_prob = torch.clamp(target_prob, min=eps)
        bias_loss = F.kl_div(input_prob, target_prob, reduction='batchmean')

        # compute robustness loss
        input_prob = F.log_softmax(rep_perturbed_g, dim=1)
        target_prob = F.softmax(rep_original_g, dim=1)
        target_prob = torch.clamp(target_prob, min=eps)
        robustness_loss = F.kl_div(input_prob, target_prob, reduction='batchmean')

        # compute total loss
        total_loss = bias_loss + args.balance * robustness_loss
        total_loss.backward()
        optimizer.step()

        # early stopping logic
        val_acc, _, val_loss = evaluate(g, features, labels, val_mask, model)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            saved_model = deepcopy(model)
            logger.info(f"Epoch {epoch:05d} | Loss {total_loss.item():.4f} | Bias {bias_loss.item():.4f} | Robustness {robustness_loss.item():.4f} | val loss {val_loss:.4f} | val acc {val_acc:.4f}")
            counter = 0                   # Reset counter if improvement
        else:
            counter += 1  
            
        if counter >= patience:
            logger.info(f"Early stopping...")
            break
    
    # evaluate the final model on the test set
    model.load_state_dict(saved_model.state_dict())
    test_acc, _, _ = evaluate(g, features, labels, test_mask, model, '{}/{}.pkl'.format(args.save_dir, args.dataset))
      
    return test_acc, saved_model


if __name__ == "__main__":
    args = TrainBuffer().parse()
    if not os.path.isdir(args.save_dir): os.makedirs(args.save_dir)
    logger = init_logger("main_logger", '{}/{}_run.log'.format(args.save_dir, args.dataset))
    analysis_logger = init_logger("analysis_logger", '{}/{}_analysis.log'.format(args.save_dir, args.dataset))

    # set the number of threads
    torch.set_num_threads(8)
    
    # initialize wandb if enabled
    if args.wandb:
        wandb.init(project="agg_buffer_exp", config={
            "id": args.wandb_id,
            "index": args.index,
            "dataset": args.dataset,
            "learning_rate": args.lr,
            "weight_decay": args.weight_decay,
            "hid_dim": args.hid_dim,
            "mp_norm": args.mp_norm,
            "dropout": args.dropout,
            "dropedge": args.dropedge,
            "balance": args.balance,
            "architecture": args.architecture
        })
        config = wandb.config

    # load data
    g = load_dataset(args.dataset, index=args.index)
    device = torch.device("cuda:{}".format(args.device))
    g = g.int().to(device)
    features = g.ndata["feat"]
    labels = g.ndata["label"]
    masks = g.ndata["train_mask"], g.ndata["val_mask"], g.ndata["test_mask"]
    
    # random seed setting
    set_seed(args.seed)

    # Initialize the extended model and original model
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
    trained_factory = {
        "SAGE": lambda: SAGE(**model_args),
        "GIN": lambda: GIN(**model_args),
        "SGC": lambda: SGC(in_size, args.hid_dim + [out_size], use_linear=args.use_linear),
        "GAT": lambda: GAT(in_size, args.hid_dim[0], out_size, use_linear=args.use_linear, dropout=args.dropout, norm=args.norm),
        "GCN": lambda: GCN(**model_args, mp_norm=args.mp_norm),
    }

    model_factory = {
        "SAGE": lambda: SAGE_B(**model_args),
        "GIN":  lambda: GIN_B(**model_args),
        "SGC":  lambda: SGC_B(in_size, args.hid_dim + [out_size], use_linear=args.use_linear, dropout=args.dropout),
        "GAT":  lambda: GAT_B(in_size, args.hid_dim[0], out_size, use_linear=args.use_linear, dropout=args.dropout, norm=args.norm),
        "GCN":  lambda: GCN_B(**model_args, mp_norm=args.mp_norm),
    }
    
    # Default to GCN if architecture not found
    trained_model = trained_factory.get(args.architecture, trained_factory["GCN"])().to(device)
    model         = model_factory.get(args.architecture, model_factory["GCN"])().to(device)

    # load pre-trained model
    pretrained_model_path = "results/base/{}.ckpt".format(args.dataset)
    if os.path.exists(pretrained_model_path):
        checkpoint = torch.load(pretrained_model_path) 
        trained_model.load_state_dict(checkpoint)
        print("Loaded pre-trained model from {}".format(pretrained_model_path))
    else:
        raise FileNotFoundError(f"Pre-trained model not found at {pretrained_model_path}")
    
    # transfer weights from the trained model to the model with buffer
    model.transfer_weights(trained_model)
    print(model)
    print("-" * 50)

    # training the buffer
    logger.info("Training...")
    time = timeit.default_timer()
    test_acc, saved_model = train(g, features, masks, model, trained_model, args)
    logger.info("Test accuracy {:.2f}| Time Consumed: {:.2f} s".format(test_acc*100, timeit.default_timer()-time))
    save_model(saved_model.state_dict(), 
               {'in_feats':in_size, 
                'hidden_lst':args.hid_dim + [out_size],
                'norm':args.norm,
                'mp_norm':args.mp_norm},
               '{}/{}'.format(args.save_dir, args.dataset))
    print("Training complete. Model saved to {}/{}.ckpt".format(args.save_dir, args.dataset))
    print("-" * 50)
    
    # evaluate the models; gnn and gnn_buffer
    print("Evaluating models...")
    model.load_state_dict(saved_model.state_dict())
    models = [(trained_model, "gnn"), (model, "gnn_buffer")]
    evaluate_and_log_models(models, g, features, labels, masks, analysis_logger, logger, args)
    print("-" * 50)
    
