import os
import logging
import json
from datetime import datetime

def setup_logging(log_dir=None, level=logging.INFO):
    logger = logging.getLogger()
    logger.setLevel(level)

    logger.handlers = []

    console_handler = logging.StreamHandler()
    console_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"math_topics_{timestamp}.log")

        file_handler = logging.FileHandler(log_file)
        file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    return logger

def create_results_dir(base_dir, model_name):
    model_name_clean = model_name.split('/')[-1]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dir_name = f"{timestamp}_{model_name_clean}"

    result_dir = os.path.join(base_dir, dir_name)
    os.makedirs(result_dir, exist_ok=True)

    return result_dir

def save_run_config(config, save_dir):
    clean_config = {}

    def clean_dict(d):
        cleaned = {}
        for k, v in d.items():
            if isinstance(v, dict):
                cleaned[k] = clean_dict(v)
            elif isinstance(v, (str, int, float, bool, list, tuple)) or v is None:
                cleaned[k] = v
            else:
                cleaned[k] = str(v)
        return cleaned

    clean_config = clean_dict(config)

    json_path = os.path.join(save_dir, 'run_config.json')
    with open(json_path, 'w') as f:
        json.dump(clean_config, f, indent=2)

    return json_path 