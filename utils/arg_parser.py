import os
import json
import argparse

class TrainGNN():
    def __init__(self, config_file = "configs/train_config.json"):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.initialize_parser()
        self.config_file = config_file
        
    def initialize_parser(self):
        self.parser.add_argument('--dataset', type=str, default='cora', \
            choices=['cora', 'citeseer', 'pubmed', 'wiki_cs', 'co_photo', 'co_computer', 'co_cs', 'co_phy', 'arxiv', 'actor', 'chameleon', 'squirrel'], help='dataset')
        self.parser.add_argument('--hid_dim', type=int, nargs='+', default=[256])
        self.parser.add_argument('--lr', type=float, default=1e-2, help='learning rate')
        self.parser.add_argument('--device', type=int, default=0, help='index of the gpu for training')
        self.parser.add_argument('--weight_decay', type=float, default=1e-4)
        self.parser.add_argument('--norm', type=str, default='identity')
        self.parser.add_argument('--save_dir', type=str, default='results/base')
        self.parser.add_argument('--mp_norm', type=str, default='right')
        self.parser.add_argument('--dropout', type=float, default=0.5)
        self.parser.add_argument('--dropedge', type=float, default=0.0)
        self.parser.add_argument('--index', type=int, default=0, help='index of the split for training')
        self.parser.add_argument('--wandb', type=bool, default=False)
        self.parser.add_argument('--wandb_id', type=int, default=0)
        self.parser.add_argument('--seed', type=int, default=42)
        self.parser.add_argument('--architecture', type=str, default='GCN')
        self.parser.add_argument('--use_linear', type=bool, default=False)

    def parse(self):
        opt = self.parser.parse_args()

        # Load configuration file
        if os.path.exists(self.config_file):
            with open(self.config_file, 'r') as f:
                config = json.load(f)

            # Override arguments based on dataset
            if opt.dataset in config:
                hyperparams = config[opt.dataset]
                for key, value in hyperparams.items():
                    setattr(opt, key, value)
        else:
            print(f"Warning: Configuration file {self.config_file} not found. Using default hyperparameters.")

        return opt
    
class TrainBuffer():
    def __init__(self, config_file = "configs/buf_config.json"):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.initialize_parser()
        self.config_file = config_file
        
    def initialize_parser(self):
        self.parser.add_argument('--dataset', type=str, default='cora', \
            choices=['cora', 'citeseer', 'pubmed', 'wiki_cs', 'co_photo', 'co_computer', 'co_cs', 'co_phy', 'arxiv', 'actor', 'chameleon', 'squirrel'], help='dataset')
        self.parser.add_argument('--hid_dim', type=int, nargs='+', default=[256])
        self.parser.add_argument('--lr', type=float, default=1e-2, help='learning rate')
        self.parser.add_argument('--device', type=int, default=0, help='index of the gpu for training')
        self.parser.add_argument('--weight_decay', type=float, default=0.0)
        self.parser.add_argument('--norm', type=str, default='identity')
        self.parser.add_argument('--save_dir', type=str, default='results/buffer')
        self.parser.add_argument('--mp_norm', type=str, default='right')
        self.parser.add_argument('--dropout', type=float, default=0.5)
        self.parser.add_argument('--dropedge', type=float, default=0.5)
        self.parser.add_argument('--index', type=int, default=0, help='index of the split for training')
        self.parser.add_argument('--wandb', type=bool, default=False)
        self.parser.add_argument('--wandb_id', type=int, default=0)
        self.parser.add_argument('--seed', type=int, default=42)
        self.parser.add_argument('--architecture', type=str, default='GCN')
        self.parser.add_argument('--use_linear', type=bool, default=False)
        # Additional hyperparameter for buffer training
        self.parser.add_argument('--balance', type=float, default=1)
        
    def parse(self):
        opt = self.parser.parse_args()

        # Load configuration file
        if os.path.exists(self.config_file):
            with open(self.config_file, 'r') as f:
                config = json.load(f)

            # Override arguments based on dataset
            if opt.dataset in config:
                hyperparams = config[opt.dataset]
                for key, value in hyperparams.items():
                    setattr(opt, key, value)
        else:
            print(f"Warning: Configuration file {self.config_file} not found. Using default hyperparameters.")

        return opt