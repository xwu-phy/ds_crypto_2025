"""
Loads and merges YAML configuration files.
"""

import os
import json
import yaml
from pathlib import Path
from typing import Dict, Any, Optional

class ConfigManager:
    def __init__(self, config_dir: Optional[str] = None):
        if config_dir is None:
            # Default to configs/ directory relative to project root
            project_root = Path(__file__).parent.parent
            self.config_dir = project_root / "configs"
        else:
            self.config_dir = Path(config_dir)
    
    def load_yaml(self, filepath: Path) -> Dict[str, Any]:
        try:
            with open(filepath, 'r') as file:
                return yaml.safe_load(file) or {}
        except FileNotFoundError:
            return {}
    
    def merge_configs(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        result = base.copy()
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self.merge_configs(result[key], value)
            else:
                result[key] = value
        return result
    
    def resolve_paths(self, config: Dict[str, Any], coin: str, granularity: str) -> Dict[str, Any]:
        """Replace {coin} and {granularity} placeholders in paths and make them absolute."""
        project_root = Path(__file__).parent.parent  # Get project root directory
        
        def resolve_templates(obj: Any, parent_key: str = "") -> Any:
            if isinstance(obj, dict):
                return {key: resolve_templates(value, key) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [resolve_templates(item, parent_key) for item in obj]
            elif isinstance(obj, str):
                # First resolve template variables
                resolved_str = obj.format(coin=coin, granularity=granularity)
                # Only make paths absolute if the key suggests it's a file/directory path
                is_path_key = any(suffix in parent_key.lower() for suffix in ['_file', '_path', '_dir', '_folder'])
                if is_path_key and not Path(resolved_str).is_absolute():
                    resolved_str = str(project_root / resolved_str)
                return resolved_str
            else:
                return obj
        result = resolve_templates(config)
        return result if isinstance(result, dict) else config
    
    def load_saved_params(self, model_type: str, config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Load saved hyperparameters from trained models."""
        try:
            if model_type == "xgboost":
                params_file = config['artifacts']['hyperparameters_file']
            elif model_type == "lstm":
                params_file = config['artifacts']['best_params_file']
            else:
                return None
                
            if Path(params_file).exists():
                with open(params_file, 'r') as f:
                    return json.load(f)
        except:
            pass
        return None
    
    def get_config(self, model: str, coin: str, granularity: str, 
                   environment: str = "development", 
                   load_saved_params: bool = False) -> Dict[str, Any]:
        """
        Load configuration for a specific model, coin, and granularity.
        
        Args:
            model: "xgboost" or "lstm"
            coin: Cryptocurrency symbol (e.g., "BTCUSDT")
            granularity: Time granularity (e.g., "1H")
            environment: "development" or "production"
            load_saved_params: Whether to load saved hyperparameters
        """
        config = self.load_yaml(self.config_dir / "base_config.yaml")
        model_config = self.load_yaml(self.config_dir / f"{model}_config.yaml")
        config = self.merge_configs(config, model_config)
        
        config['data']['default_coin'] = coin
        config['data']['granularity'] = granularity
        
        # Apply environment-specific settings
        if 'hyperparameter_optimization' in config and environment in config['hyperparameter_optimization']:
            env_settings = config['hyperparameter_optimization'][environment]
            config['hyperparameter_optimization'].update(env_settings)
        
        # Resolve path templates
        config = self.resolve_paths(config, coin, granularity)
        
        # Load saved hyperparameters if requested
        if load_saved_params:
            saved_params = self.load_saved_params(model, config)
            if saved_params:
                config['saved_hyperparameters'] = saved_params
        
        return config

def get_model_config(model: str, coin: str, granularity: str, 
                    environment: str = "development",
                    load_saved_params: bool = False) -> Dict[str, Any]:
    manager = ConfigManager()
    return manager.get_config(model, coin, granularity, environment, load_saved_params)

def get_base_config() -> Dict[str, Any]:
    manager = ConfigManager()
    return manager.load_yaml(manager.config_dir / "base_config.yaml")