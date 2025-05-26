import re
import numpy as np
import argparse
import os

def parse_log(file_path, n):
    with open(file_path, 'r') as f:
        lines = f.readlines()

    # Regex to extract Degree Bias data
    degree_pattern = re.compile(
        r"Degree Bias\s+\|\s+(gnn|gnn_buffer)\s+\|\s+Overall:\s+Acc\s+(\d+\.\d+)%,\s+F1\s+(\d+\.\d+)\s+\|\s+Head:\s+Acc\s+(\d+\.\d+)%,\s+F1\s+(\d+\.\d+)\s+\|\s+Tail:\s+Acc\s+(\d+\.\d+)%,\s+F1\s+(\d+\.\d+)"
    )
    # Regex to extract Structural Disparity data
    homophily_pattern = re.compile(
        r"Structural Disparity\s+\|\s+(gnn|gnn_buffer)\s+\|\s+Overall:\s+Acc\s+(\d+\.\d+)%,\s+F1\s+(\d+\.\d+)\s+\|\s+Homophilous:\s+Acc\s+(\d+\.\d+)%,\s+F1\s+(\d+\.\d+)\s+\|\s+Heterophilous:\s+Acc\s+(\d+\.\d+)%,\s+F1\s+(\d+\.\d+)"
    )

    degree_results = []
    homophily_results = []

    for line in lines:
        degree_match = degree_pattern.search(line)
        if degree_match:
            model_type = degree_match.group(1)
            data = list(map(float, degree_match.groups()[1:]))
            degree_results.append((model_type, data))
        
        homophily_match = homophily_pattern.search(line)
        if homophily_match:
            model_type = homophily_match.group(1)
            data = list(map(float, homophily_match.groups()[1:]))
            homophily_results.append((model_type, data))

    # Filter last n results for each type
    degree_gnn_results = [data for model, data in degree_results if model == "gnn"][-n:]
    degree_gnn_buffer_results = [data for model, data in degree_results if model == "gnn_buffer"][-n:]

    homophily_gnn_results = [data for model, data in homophily_results if model == "gnn"][-n:]
    homophily_gnn_buffer_results = [data for model, data in homophily_results if model == "gnn_buffer"][-n:]

    return degree_gnn_results, degree_gnn_buffer_results, homophily_gnn_results, homophily_gnn_buffer_results

def compute_stats(data):
    if len(data) == 0:
        return None, None
    data_array = np.array(data)
    means = np.mean(data_array, axis=0)
    stds = np.std(data_array, axis=0)
    return means, stds

def main():
    parser = argparse.ArgumentParser(description="Analyze log results for a given dataset.")
    parser.add_argument("--dataset", type=str, required=True, help="Name of the dataset.")
    parser.add_argument("--repeats", type=int, required=True, help="Number of most recent results to consider.")
    args = parser.parse_args()

    log_file = f"results/buffer/{args.dataset}_analysis.log"
    
    # Check if log file exists
    if not os.path.exists(log_file):
        print(f"Error: Log file '{log_file}' not found.")
        return

    # Parse the log and calculate stats
    degree_gnn, degree_gnn_buffer, homophily_gnn, homophily_gnn_buffer = parse_log(log_file, args.repeats)

    if not degree_gnn or not degree_gnn_buffer or not homophily_gnn or not homophily_gnn_buffer:
        print(f"Error: Not enough data in the log file for the last {args.repeats} results.")
        return

    # Compute stats for Degree Bias
    degree_gnn_means, degree_gnn_stds = compute_stats(degree_gnn)
    degree_gnn_buffer_means, degree_gnn_buffer_stds = compute_stats(degree_gnn_buffer)

    # Compute stats for Structural Disparity
    homophily_gnn_means, homophily_gnn_stds = compute_stats(homophily_gnn)
    homophily_gnn_buffer_means, homophily_gnn_buffer_stds = compute_stats(homophily_gnn_buffer)

    print(f"Summary for dataset: {args.dataset} (last {args.repeats} entries)")

    # Degree Bias summary
    print("Degree Bias          | gnn        | "
          f"Overall:  Acc {degree_gnn_means[0]:6.2f}% ± {degree_gnn_stds[0]:5.2f} | "
          f"Head:        Acc {degree_gnn_means[2]:6.2f}% ± {degree_gnn_stds[2]:5.2f} | "
          f"Tail:          Acc {degree_gnn_means[4]:6.2f}% ± {degree_gnn_stds[4]:5.2f}")
    print("Degree Bias          | gnn_buffer | "
          f"Overall:  Acc {degree_gnn_buffer_means[0]:6.2f}% ± {degree_gnn_buffer_stds[0]:5.2f} | "
          f"Head:        Acc {degree_gnn_buffer_means[2]:6.2f}% ± {degree_gnn_buffer_stds[2]:5.2f} | "
          f"Tail:          Acc {degree_gnn_buffer_means[4]:6.2f}% ± {degree_gnn_buffer_stds[4]:5.2f}")

    # Structural Disparity summary
    print("Structural Disparity | gnn        | "
          f"Overall:  Acc {homophily_gnn_means[0]:6.2f}% ± {homophily_gnn_stds[0]:5.2f} | "
          f"Homophilous: Acc {homophily_gnn_means[2]:6.2f}% ± {homophily_gnn_stds[2]:5.2f} | "
          f"Heterophilous: Acc {homophily_gnn_means[4]:6.2f}% ± {homophily_gnn_stds[4]:5.2f}")
    print("Structural Disparity | gnn_buffer | "
          f"Overall:  Acc {homophily_gnn_buffer_means[0]:6.2f}% ± {homophily_gnn_buffer_stds[0]:5.2f} | "
          f"Homophilous: Acc {homophily_gnn_buffer_means[2]:6.2f}% ± {homophily_gnn_buffer_stds[2]:5.2f} | "
          f"Heterophilous: Acc {homophily_gnn_buffer_means[4]:6.2f}% ± {homophily_gnn_buffer_stds[4]:5.2f}")
    print("-" * 50)
    
if __name__ == "__main__":
    main()