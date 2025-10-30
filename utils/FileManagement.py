import os

def check_directory(directory_path: str):
    """
    Checks if the directory for the given filename exists.
    If not, it creates the necessary directories.
    """
    if not os.path.exists(directory_path):
        os.makedirs(directory_path, exist_ok=True)
        print(f"Created directory: {directory_path}")

def get_next_filename(directory, prefix, format, coconut_type):
    """
    Calculates the full path for the next sequentially numbered audio file.
    Also ensures the target directory exists before starting the search.
    """
    check_directory(directory)

    counter = 1
    
    # Loop until we find a filename that doesn't exist
    while True:
        filename = f"{prefix}{coconut_type}{counter:04d}{format}"
        full_path = os.path.join(directory, filename)
        
        # If the file path does not exist, this is our file name
        if not os.path.exists(full_path):
            return full_path
        
        counter += 1
        print(f"Checked {full_path}, it exists. Trying next...")

