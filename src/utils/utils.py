import json

def read_txt(file_path, encoding="utf-8"):
    """
    Reads a text file and returns its content as a string.

    Args:
        file_path (str): Path to the text file.
        encoding (str): Encoding type (default: 'utf-8').

    Returns:
        str: Content of the file.
    """
    with open(file_path, "r", encoding=encoding) as file:
        return file.read()



def read_json(file_path, encoding="utf-8"):
    """
    Reads a JSON file and returns its contents as a dictionary.

    Args:
        file_path (str): The path to the JSON file.

    Returns:
        dict: The JSON data loaded as a dictionary.
    """
    try:
        with open(file_path, "r", encoding=encoding) as file:
            data = json.load(file)
        return data
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return {}
    except json.JSONDecodeError:
        print(f"Error: Failed to decode JSON from '{file_path}'.")
        return {}