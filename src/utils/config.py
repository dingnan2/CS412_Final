"""
Configuration management for CS 412 Research Project
"""

import yaml
import os
from pathlib import Path
from typing import Dict, Any


class Config:
    """Configuration manager for the project"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize configuration from YAML file"""
        self.config_path = config_path
        self.config = self._load_config()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        try:
            with open(self.config_path, 'r') as file:
                return yaml.safe_load(file)
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing configuration file: {e}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key (supports dot notation)"""
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
                
        return value
    
    def get_data_path(self, data_type: str) -> str:
        """Get data path for specific data type"""
        base_path = self.get(f"data.{data_type}_path")
        return os.path.join(os.getcwd(), base_path)
    
    def get_model_params(self, model_name: str) -> Dict[str, Any]:
        """Get model parameters for specific model"""
        return self.get(f"models.{model_name}", {})
    
    def get_feature_params(self, feature_type: str) -> Dict[str, Any]:
        """Get feature engineering parameters"""
        return self.get(f"features.{feature_type}", {})
    
    def get_evaluation_params(self) -> Dict[str, Any]:
        """Get evaluation parameters"""
        return self.get("evaluation", {})
    
    def get_output_path(self, output_type: str) -> str:
        """Get output path for specific output type"""
        base_path = self.get(f"output.{output_type}_path")
        return os.path.join(os.getcwd(), base_path)
    
    def create_directories(self):
        """Create necessary directories if they don't exist"""
        directories = [
            self.get_data_path("raw"),
            self.get_data_path("processed"),
            self.get_data_path("samples"),
            self.get_output_path("models"),
            self.get_output_path("plots"),
            self.get_output_path("reports")
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)


# Global configuration instance
config = Config()
