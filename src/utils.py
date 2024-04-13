import logging
import yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

def load_config(config_file):
    with open(config_file, "rb") as f:
        config = yaml.safe_load(f)
    return config