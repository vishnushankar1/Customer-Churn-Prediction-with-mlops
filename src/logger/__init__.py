# src/logger/__init__.py

import logging
import os
from datetime import datetime

# Use current working directory (not from_root!)
ROOT_PATH = os.getcwd()

# Create log directory and file
LOG_DIR = "logs"
LOG_FILE = f"{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.log"
log_dir_path = os.path.join(ROOT_PATH, LOG_DIR)
os.makedirs(log_dir_path, exist_ok=True)

log_file_path = os.path.join(log_dir_path, LOG_FILE)
print("📝 Log file path:", log_file_path)

def configure_logger():
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    # Clear any old handlers
    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter("[ %(asctime)s ] %(name)s - %(levelname)s - %(message)s")

    # ✅ File handler
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG)

    # ✅ Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)

    # Add both handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

# Call it immediately
configure_logger()
