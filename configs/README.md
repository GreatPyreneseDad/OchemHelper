# Configuration Management

This directory contains configuration files for the OChem Helper platform.

## Configuration Files

- **default.yaml**: Base configuration with default values
- **development.yaml**: Development environment overrides
- **production.yaml**: Production environment settings
- **test.yaml**: Test environment configuration

## Configuration Hierarchy

1. **Base Configuration**: `default.yaml` contains all default settings
2. **Environment Overrides**: Environment-specific files override base values
3. **Environment Variables**: Can override any configuration value

## Usage

### Loading Configuration

```python
from core.config import load_config, get_config

# Load with specific environment
config = load_config(environment="production")

# Access configuration values
api_host = config.api.host
vae_dim = config.models.vae.latent_dim

# Use dot notation
max_atoms = config.get('chemistry.max_atoms')
```

### Environment Variables

Set the environment:
```bash
export OCHEM_ENV=production
```

Override specific values:
```bash
export JWT_SECRET="your-secret-key"
export DATABASE_URL="postgresql://user:pass@host/db"
```

## Configuration Structure

### Models Configuration
- **vae**: VAE model parameters
- **predictor**: Property prediction settings
- **generator**: Molecule generation settings

### API Configuration
- Host, port, and worker settings
- CORS and rate limiting
- Authentication settings

### Training Configuration
- Batch size and learning rate
- Checkpoint and logging directories
- Early stopping parameters

### Data Configuration
- Data directory paths
- Dataset splits
- Cache settings

### MCP Configuration
- Server settings
- Request handling
- Caching parameters

### Chemistry Configuration
- Molecular constraints
- Drug-likeness rules
- Allowed elements

## Adding New Configuration

1. Add to `default.yaml`:
```yaml
my_feature:
  enabled: true
  option1: "value"
  option2: 42
```

2. Override in environment files as needed:
```yaml
# production.yaml
my_feature:
  enabled: true
  option1: "prod-value"
```

3. Access in code:
```python
config = get_config()
if config.my_feature.enabled:
    value = config.my_feature.option1
```

## Environment Variables

Use `${VAR_NAME}` syntax in YAML:
```yaml
database:
  url: "${DATABASE_URL}"
  password: "${DB_PASSWORD}"
```

## Best Practices

1. **Secrets**: Never commit secrets. Use environment variables
2. **Defaults**: Provide sensible defaults in `default.yaml`
3. **Documentation**: Comment complex configuration options
4. **Validation**: Validate configuration on load
5. **Type Safety**: Use consistent types for values

## Configuration Validation

The configuration loader validates:
- Required fields are present
- Types are correct
- Values are within valid ranges

## Dynamic Configuration

Update configuration at runtime:
```python
from core.config import update_config

update_config({
    'api': {'port': 8080},
    'features': {'enable_caching': False}
})
```

## Testing Configuration

For tests, use the test configuration:
```python
config = load_config(environment="test")
```

This provides:
- Smaller model sizes
- Local file paths
- Disabled external services
- Fixed random seeds