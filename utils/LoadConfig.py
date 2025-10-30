import json
import os

def load_config_json(config_filename='config.json'):
    """
    Reads configuration from the specified JSON file by reliably resolving 
    the absolute path relative to the module's location.
    """
    # 1. Get the directory of the *current* module (e.g., /project_root/utils/)
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # 2. Go up one directory level to reach the project root (e.g., /project_root/)
    project_root = os.path.dirname(current_dir)

    # 3. Join the root directory with the config filename
    config_file_path = os.path.join(project_root, config_filename)
    
    # 4. Check and load the config using the absolute path
    if not os.path.exists(config_file_path):
        raise FileNotFoundError(f"Configuration file not found at: {config_file_path}")
    
    with open(config_file_path, 'r') as f:
        # The json.load() function reads the file and returns a Python dictionary
        config_data = json.load(f)
        
    return config_data

# The final, crucial step: load the config into a variable 
# so it's accessible when the module is imported.
CONFIG = load_config_json()