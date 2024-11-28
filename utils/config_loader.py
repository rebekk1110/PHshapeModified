import yaml
from pathlib import Path
import os

def load_config(config_path):
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config
    except FileNotFoundError:
        raise FileNotFoundError(f"Config file not found: {config_path}")
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing YAML file: {e}")

def get_full_path(base_path, relative_path):
    return str(Path(base_path) / relative_path)

def process_config(config):
    base_path = Path.cwd()
    
    # Process input paths
    config['data']['input']['raster_folder'] = get_full_path(base_path, config['data']['output']['out_tif_tiles_folder'])
    
    # Process output paths
    for key in config['data']['output']:
        config['data']['output'][key] = get_full_path(base_path, config['data']['output'][key])
    
    return config

def get_config(config_path=None):
    if config_path is None:
        config_path = os.environ.get('CONFIG_PATH', Path(__file__).parent.parent / 'config' / 'config_raster.yaml')
    return process_config(load_config(config_path))