import json
import csv
import os


def write_to_txt(file_path, predictions_list):
    """Writes prediction results to a TXT file."""
    with open(file_path, "w", encoding="utf-8") as f:

        for prediction in predictions_list:
            for key, value in prediction.items():
                f.write(f"{key}: {value}\n")
                f.write("-" * 24 + "\n")
            # f.write("-" * 48 + "\n\n")
            f.write("\n\n")

def write_to_json(file_path, predictions_list):
    """Writes prediction results to a JSON file."""
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(predictions_list, f, ensure_ascii=False, indent=4)

def write_to_csv(file_path, predictions_list):
    """Writes prediction results to a CSV file."""
    fieldnames = predictions_list[0].keys()
    with open(file_path, mode='w', newline='', encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(predictions_list)


def save_predictions(predictions_list, directory, filename):
    """
    Saves predictions in a format determined by the file extension.

    Args:
        predictions_list (list): List of prediction results.
        directory (str): Directory path to save files.
        filename (str): Filename with extension (e.g., 'results.txt', 'results.json', 'results.csv').
    """

    # Ensure directory exists
    os.makedirs(directory, exist_ok=True)

    # Extract file extension
    file_extension = filename.split('.')[-1].lower()
    file_path = os.path.join(directory, filename)

    # Choose appropriate write function
    if file_extension == "txt":
        write_to_txt(file_path, predictions_list)
    elif file_extension == "json":
        write_to_json(file_path, predictions_list)
    elif file_extension == "csv":
        write_to_csv(file_path, predictions_list)
    else:
        raise ValueError("Unsupported file extension. Use '.txt', '.json', or '.csv'.")


def save_metrics(metrics, directory, filename):
    """
    Saves evaluation metrics (e.g., accuracy) to a TXT file.

    Args:
        metrics (dict): Dictionary containing evaluation metrics.
        directory (str): Directory path to save the file.
        filename (str):  Filename with extension .txt
    """
    file_path = os.path.join(directory, filename)

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(f"Experiment Name: {metrics.get('exp_name', 'N/A')}\n")
        f.write(f"Experiment Variant: {metrics.get('exp_variant', 'N/A')}\n")
        f.write("-" * 48 + "\n\n")
        
        for key, value in metrics.items():
            if key != "exp_name" and key != "exp_variant":  # Avoid duplicating experiment name
                f.write(f"{key.capitalize()}: {value}\n")
        
        f.write("-" * 48 + "\n")

