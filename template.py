import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s:')


project_name = "Bioneural"

list_of_files = [
    ".github/workflows/.gitkeep",
    f"src/{project_name}/__init__.py",
    f"src/{project_name}/core/__init__.py",
    f"src/{project_name}/core/homeostasis.py",
    f"src/{project_name}/core/biololgicallayer.py",
    f"src/{project_name}/metrics/__init__.py",
    f"src/{project_name}/metrics/plotting.py",
    f"src/{project_name}/metrics/healtracker.py",
    f"src/{project_name}/utils/logging.py",
    f"src/{project_name}/visualization/biosysvisualization.py",
    f"src/{project_name}__init__.py",
    
    "requirements.txt",
    "setup.py",

]
for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)

    if filedir != "":
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"Creating directory:{filedir} for the file {filename}")

    
    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath,'w') as f:
            pass
            logging.info(f"Creating empty file: {filepath}")


    
    else:
        logging.info(f"{filename} is already exists")