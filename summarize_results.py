import os
import re
import pandas as pd

def summarize_results(exps_dir="exps/", output_csv="results_summary.csv"):
    """
    Reads results from the '{exps_dir}/{exp_name}/results/test_metrics.txt' directories, 
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
        results_file = os.path.join(exps_dir, exp_name, "results", "test_metrics.txt")
        
        if os.path.exists(results_file):
            with open(results_file, "r", encoding="utf-8") as f:
                content = f.read()

                # Extract accuracy using regex
                accuracy_match = re.search(r"Accuracy: \{'accuracy': ([\d\.]+)\}", content)
                accuracy = float(accuracy_match.group(1)) if accuracy_match else None

                # Store results
                experiments.append({
                    "Experiment": exp_name,
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


summarize_results(exps_dir="/exps/", output_csv="results_summary.csv")