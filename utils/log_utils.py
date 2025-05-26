import sys
import timeit
import wandb
import logging
from evaluate.utils import evaluate, evaluate, evaluate_degree_bias, evaluate_edge_removal, evaluate_structural_disparity

# Initialize logger
def init_logger(name, filename=None):
    # Create a logger with a unique name
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Avoid duplicate handlers
    if not logger.hasHandlers():
        # Create console handler
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(logging.Formatter(
            fmt="[%(asctime)s] {%(name)s} %(levelname)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S"
        ))
        logger.addHandler(stream_handler)

        # Optionally add file handler
        if filename is not None:
            file_handler = logging.FileHandler(filename=filename)
            file_handler.setFormatter(logging.Formatter(
                fmt="[%(asctime)s] {%(name)s} %(levelname)s - %(message)s",
                datefmt="%m/%d/%Y %H:%M:%S"
            ))
            logger.addHandler(file_handler)

    return logger

def log_degree_bias(analysis_logger, model_name, results, wandb_enabled):
    msg = (
        f"{'Degree Bias':<20} | {model_name:<10} | "
        f"{'Overall:':<9} Acc {results['overall']['accuracy'] * 100:6.2f}%, "
        f"F1 {results['overall']['f1'] * 100:6.2f} | "
        f"{'Head:':<12} Acc {results['head']['accuracy'] * 100:6.2f}%, "
        f"F1 {results['head']['f1'] * 100:6.2f} | "
        f"{'Tail:':<14} Acc {results['tail']['accuracy'] * 100:6.2f}%, "
        f"F1 {results['tail']['f1'] * 100:6.2f}"
    )
    analysis_logger.info(msg)
    # Log to wandb if enabled
    if wandb_enabled:
        wandb.log({
            f"{model_name}_head_acc": results['head']['accuracy'] * 100,
            f"{model_name}_tail_acc": results['tail']['accuracy'] * 100
        })

def log_structural_disparity(analysis_logger, model_name, results, wandb_enabled):
    msg = (
        f"{'Structural Disparity':<20} | {model_name:<10} | "
        f"{'Overall:' :<9} Acc {results['overall']['accuracy'] * 100:6.2f}%, "
        f"F1 {results['overall']['f1'] * 100:6.2f} | "
        f"{'Homophilous:':<12} Acc {results['homophilous']['accuracy'] * 100:6.2f}%, "
        f"F1 {results['homophilous']['f1'] * 100:6.2f} | "
        f"{'Heterophilous:':<14} Acc {results['heterophilous']['accuracy'] * 100:6.2f}%, "
        f"F1 {results['heterophilous']['f1'] * 100:6.2f}"
    )
    analysis_logger.info(msg)
    # Log to wandb if enabled
    if wandb_enabled:
        wandb.log({
            f"{model_name}_homophilous_acc": results['homophilous']['accuracy'] * 100,
            f"{model_name}_heterophilous_acc": results['heterophilous']['accuracy'] * 100
        })


def log_edge_removal(logger, model_name, results, wandb_enabled):
    steps = [100, 75, 50, 25, 0]
    acc_strs = [
        f"{step:>3}: {results[step]['accuracy'] * 100:6.2f}%" for step in steps if step in results
    ]
    msg = f"{'Edge Removal':<15} | {model_name:<12} | " + " | ".join(acc_strs)
    logger.info(msg)
    # Log to wandb if enabled
    if wandb_enabled:
        for step, metrics in results.items():
            wandb.log({
                f"{model_name}_removal_{step}_acc": metrics['accuracy'] * 100,
            })


def evaluate_and_log_models(models, g, features, labels, masks, analysis_logger, logger, args, logits = None, repeats = 5):
    val_mask = masks[1]
    test_mask = masks[2]
    degrees = g.in_degrees()
    for model_i, model_name in models:
        val_acc, _, val_loss = evaluate(g, features, labels, val_mask, model_i)
        test_acc, _, test_loss = evaluate(g, features, labels, test_mask, model_i)
        # Log to wandb if enabled
        if args.wandb:
            wandb.log({
                f"{model_name}_val_acc": val_acc * 100,
                f"{model_name}_test_acc": test_acc * 100
            })

        results = evaluate_degree_bias(
            g=g,
            features=features,
            labels=labels,
            mask=test_mask,
            model=model_i,
            degrees=degrees,
        )
        log_degree_bias(analysis_logger, model_name, results, args.wandb)

        results = evaluate_structural_disparity(
            g=g,
            features=features,
            labels=labels,
            mask=test_mask,
            model=model_i,
        )
        log_structural_disparity(analysis_logger, model_name, results, args.wandb)

        '''
        results = evaluate_edge_removal(
            g=g,
            features=features,
            labels=labels,
            mask=test_mask,
            model=model_i,
            repeats= repeats
        )
        log_edge_removal(analysis_logger, model_name, results, args.wandb)
        '''
        