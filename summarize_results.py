import os
import re
import pandas as pd

def summarize_results(exps_dir="exps/", output_csv="results_summary.csv"):
    """
    Reads results from '{exps_dir}/{exp_name}/results/test_metrics_*.txt' files, 
    extracts accuracy scores, saves them to a CSV file, and displays them in Markdown format.

    Args:
        exps_dir (str): The base directory containing experiment results.
        output_csv (str): The name of the output CSV file.
    """
    experiments = []
    results_dir = os.path.join(exps_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)

    # Iterate through experiment directories
    for exp_name in os.listdir(exps_dir):
        for exp_variant in os.listdir(os.path.join(exps_dir, exp_name)):
            exp_results_dir = os.path.join(exps_dir, exp_name, exp_variant, "results")
            
            if os.path.isdir(exp_results_dir):  # Ensure it's a directory
                # Find all test_metrics_*.txt files
                for results_file in os.listdir(exp_results_dir):
                    if results_file.startswith("test_metrics") and results_file.endswith(".txt"):
                        results_file_path = os.path.join(exp_results_dir, results_file)
                        
                        if os.path.exists(results_file_path):
                            with open(results_file_path, "r", encoding="utf-8") as f:
                                content = f.read()

                                # Extract accuracy using regex
                                accuracy_match = re.search(r"Accuracy: \{'accuracy': ([\d\.]+)\}", content)
                                accuracy = float(accuracy_match.group(1)) if accuracy_match else None

                                # Store results
                                experiments.append({
                                    "Experiment Name": exp_name,
                                    "Experiment Variant": exp_variant,
                                    "Results File": results_file,
                                    "Accuracy": accuracy,
                                })

    # Convert to Pandas DataFrame
    df_results = pd.DataFrame(experiments)

    # Save to CSV
    df_results.to_csv(os.path.join(results_dir, output_csv), index=False)
    print(f"✅ Results saved to {output_csv}")

    # Show Markdown table
    markdown_table = df_results.to_markdown(index=False)
    print("\n📊 Experiment Results\n")
    print(markdown_table)

    return df_results  # Return DataFrame for further analysis


summarize_results(exps_dir="exps_modal", output_csv="results_summary.csv")
