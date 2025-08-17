"""Configuration management for OChem Helper."""

import os
import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass, field
import json

logger = logging.getLogger(__name__)


@dataclass
class Config:
    """Configuration container with dot notation access."""
    
    def __init__(self, data: Dict[str, Any]):
        """Initialize configuration from dictionary."""
        for key, value in data.items():
            if isinstance(value, dict):
                setattr(self, key, Config(value))
            else:
                setattr(self, key, value)
    
    def __getattr__(self, name: str) -> Any:
        """Get attribute with default None."""
        return self.__dict__.get(name)
    
    def __setattr__(self, name: str, value: Any) -> None:
        """Set attribute."""
        self.__dict__[name] = value
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get value with default."""
        keys = key.split('.')
        value = self
        
        for k in keys:
            if hasattr(value, k):
                value = getattr(value, k)
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any) -> None:
        """Set value using dot notation."""
        keys = key.split('.')
        obj = self
        
        for k in keys[:-1]:
            if not hasattr(obj, k):
                setattr(obj, k, Config({}))
            obj = getattr(obj, k)
        
        setattr(obj, keys[-1], value)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {}
        
        for key, value in self.__dict__.items():
            if isinstance(value, Config):
                result[key] = value.to_dict()
            else:
                result[key] = value
        
        return result
    
    def update(self, other: Union[Dict[str, Any], 'Config']) -> None:
        """Update configuration with another config or dict."""
        if isinstance(other, Config):
            other = other.to_dict()
        
        for key, value in other.items():
            if isinstance(value, dict) and hasattr(self, key):
                existing = getattr(self, key)
                if isinstance(existing, Config):
                    existing.update(value)
                else:
                    setattr(self, key, Config(value))
            else:
                if isinstance(value, dict):
                    setattr(self, key, Config(value))
                else:
                    setattr(self, key, value)


class ConfigLoader:
    """Load and manage configurations."""
    
    def __init__(self, config_dir: Optional[str] = None):
        """Initialize configuration loader.
        
        Args:
            config_dir: Directory containing config files
        """
        if config_dir is None:
            # Default to configs directory relative to this file
            config_dir = Path(__file__).parent.parent.parent / "configs"
        
        self.config_dir = Path(config_dir)
        self._cache = {}
    
    def load(self, config_name: str = "default", 
             environment: Optional[str] = None) -> Config:
        """Load configuration.
        
        Args:
            config_name: Base configuration name
            environment: Environment-specific config to overlay
            
        Returns:
            Config object
        """
        # Check cache
        cache_key = f"{config_name}:{environment}"
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        # Load base configuration
        base_config = self._load_yaml(f"{config_name}.yaml")
        
        # Load environment-specific configuration if specified
        if environment:
            env_config = self._load_yaml(f"{environment}.yaml")
            if env_config:
                base_config = self._merge_configs(base_config, env_config)
        
        # Apply environment variable overrides
        base_config = self._apply_env_overrides(base_config)
        
        # Create Config object
        config = Config(base_config)
        
        # Cache result
        self._cache[cache_key] = config
        
        return config
    
    def _load_yaml(self, filename: str) -> Dict[str, Any]:
        """Load YAML configuration file."""
        filepath = self.config_dir / filename
        
        if not filepath.exists():
            logger.warning(f"Configuration file not found: {filepath}")
            return {}
        
        try:
            with open(filepath, 'r') as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            logger.error(f"Error loading configuration {filepath}: {e}")
            return {}
    
    def _merge_configs(self, base: Dict[str, Any], 
                      overlay: Dict[str, Any]) -> Dict[str, Any]:
        """Merge two configuration dictionaries."""
        result = base.copy()
        
        for key, value in overlay.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_configs(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def _apply_env_overrides(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Apply environment variable overrides."""
        return self._process_dict_env_vars(config)
    
    def _process_dict_env_vars(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively process dictionary for environment variables."""
        result = {}
        
        for key, value in data.items():
            if isinstance(value, dict):
                result[key] = self._process_dict_env_vars(value)
            elif isinstance(value, str) and value.startswith("${") and value.endswith("}"):
                # Environment variable reference
                env_var = value[2:-1]
                env_value = os.environ.get(env_var)
                
                if env_value is not None:
                    # Try to parse as JSON first
                    try:
                        result[key] = json.loads(env_value)
                    except:
                        # Use as string
                        result[key] = env_value
                else:
                    logger.warning(f"Environment variable {env_var} not found")
                    result[key] = None
            else:
                result[key] = value
        
        return result
    
    def save(self, config: Config, filename: str) -> None:
        """Save configuration to file."""
        filepath = self.config_dir / filename
        
        try:
            with open(filepath, 'w') as f:
                yaml.dump(config.to_dict(), f, default_flow_style=False)
            logger.info(f"Configuration saved to {filepath}")
        except Exception as e:
            logger.error(f"Error saving configuration: {e}")
    
    def validate(self, config: Config, schema: Optional[Dict[str, Any]] = None) -> bool:
        """Validate configuration against schema."""
        if schema is None:
            # Basic validation - check required fields
            required_fields = [
                'models.vae.vocab_size',
                'api.host',
                'api.port',
                'data.raw_dir'
            ]
            
            for field in required_fields:
                if config.get(field) is None:
                    logger.error(f"Required configuration field missing: {field}")
                    return False
        
        return True


# Global configuration instance
_config_loader = ConfigLoader()
_global_config = None


def load_config(config_name: str = "default", 
                environment: Optional[str] = None,
                force_reload: bool = False) -> Config:
    """Load global configuration.
    
    Args:
        config_name: Base configuration name
        environment: Environment-specific config
        force_reload: Force reload even if cached
        
    Returns:
        Config object
    """
    global _global_config
    
    if _global_config is None or force_reload:
        # Check environment variable
        if environment is None:
            environment = os.environ.get("OCHEM_ENV", "development")
        
        _global_config = _config_loader.load(config_name, environment)
        
        # Validate configuration
        if not _config_loader.validate(_global_config):
            raise ValueError("Invalid configuration")
        
        logger.info(f"Loaded configuration: {config_name} with environment: {environment}")
    
    return _global_config


def get_config() -> Config:
    """Get current global configuration."""
    if _global_config is None:
        return load_config()
    return _global_config


def update_config(updates: Dict[str, Any]) -> None:
    """Update global configuration."""
    global _global_config
    
    if _global_config is None:
        _global_config = load_config()
    
    _global_config.update(updates)
    logger.info("Configuration updated")


# Convenience functions
def get_model_config(model_name: str) -> Config:
    """Get model-specific configuration."""
    config = get_config()
    return getattr(config.models, model_name, Config({}))


def get_api_config() -> Config:
    """Get API configuration."""
    return get_config().api


def get_data_config() -> Config:
    """Get data configuration."""
    return get_config().data


def get_training_config() -> Config:
    """Get training configuration."""
    return get_config().training


def get_chemistry_config() -> Config:
    """Get chemistry configuration."""
    return get_config().chemistry


# Example usage
if __name__ == "__main__":
    # Load development configuration
    config = load_config(environment="development")
    
    # Access configuration values
    print(f"API Host: {config.api.host}")
    print(f"API Port: {config.api.port}")
    print(f"VAE Latent Dim: {config.models.vae.latent_dim}")
    
    # Use dot notation
    print(f"Max atoms: {config.get('chemistry.max_atoms')}")
    
    # Update configuration
    config.set('api.port', 8080)
    print(f"Updated port: {config.api.port}")
    
    # Convert to dict
    config_dict = config.to_dict()
    print(f"Config keys: {list(config_dict.keys())}")