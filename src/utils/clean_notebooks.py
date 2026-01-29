import json
import argparse
from pathlib import Path


def clean_notebook(notebook_path):
    """
    Remove all outputs and execution counts from a Jupyter notebook.
    
    Args:
        notebook_path (str or Path): Path to the Jupyter notebook file
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        notebook_path = Path(notebook_path)
        
        if not notebook_path.exists():
            print(f"File not found: {notebook_path}")
            return False
        
        if notebook_path.suffix != '.ipynb':
            print(f"Not a Jupyter notebook: {notebook_path}")
            return False
        
        # Load notebook
        with open(notebook_path, 'r') as f:
            nb = json.load(f)
        
        # Clean outputs and execution counts
        for cell in nb['cells']:
            cell['outputs'] = []
            if 'execution_count' in cell:
                cell['execution_count'] = None
        
        # Save cleaned notebook
        with open(notebook_path, 'w') as f:
            json.dump(nb, f, indent=1)
        
        print(f"Cleaned: {notebook_path}")
        return True
        
    except json.JSONDecodeError:
        print(f"Invalid JSON in: {notebook_path}")
        return False
    except Exception as e:
        print(f"Error processing {notebook_path}: {e}")
        return False


def clean_notebooks_in_directory(directory, recursive=True):
    """
    Clean all Jupyter notebooks in a directory.
    
    Args:
        directory (str or Path): Directory to search
        recursive (bool): If True, search subdirectories recursively
        
    Returns:
        tuple: (total_processed, total_cleaned)
    """
    directory = Path(directory)
    
    if not directory.exists():
        print(f"Directory not found: {directory}")
        return 0, 0
    
    pattern = '**/*.ipynb' if recursive else '*.ipynb'
    notebooks = list(directory.glob(pattern))
    
    if not notebooks:
        print(f"No notebooks found in: {directory}")
        return 0, 0
    
    total_cleaned = 0
    for notebook in notebooks:
        if clean_notebook(notebook):
            total_cleaned += 1
    
    return len(notebooks), total_cleaned


def main():
    """Command line interface for notebook cleaning."""
    parser = argparse.ArgumentParser(
        description='Clean Jupyter notebook outputs'
    )
    parser.add_argument(
        'path',
        help='Path to notebook file or directory'
    )
    parser.add_argument(
        '-r', '--recursive',
        action='store_true',
        help='Search recursively in subdirectories (for directory paths)'
    )
    
    args = parser.parse_args()
    path = Path(args.path)
    
    if path.is_file():
        clean_notebook(path)
    elif path.is_dir():
        total, cleaned = clean_notebooks_in_directory(path, recursive=args.recursive)
        print(f"\nSummary: {cleaned}/{total} notebooks cleaned")
    else:
        print(f"Invalid path: {path}")


if __name__ == '__main__':
    main()
