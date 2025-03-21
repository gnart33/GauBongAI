# Data Processing

The `DataProcessingManager` is the central component for data processing in GauBongAI. 
It provides a unified interface for:

- Loading data from various file formats using plugins
- Processing data through configurable pipelines
- Storing and retrieving processed data

## Basic Usage

```python
from gaubongai.data_management import DataProcessingManager

manager = DataProcessingManager()

# Process a CSV file
data_info = manager.process_file("data.csv")

# Process with transformation pipeline
data_info = manager.process_file("data.csv", pipeline_name="normalize")
```

6. Update any configuration or dependency files:
```toml:pyproject.toml
[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]

[tool.coverage.run]
source = ["gaubongai"]
omit = [
    "tests/*",
    "gaubongai/data_management/processing.py",  # Updated path
]
```

7. Consider adding a deprecation warning if this is a public API:
```python:gaubongai/data_management/ingestion.py
import warnings

class DataIngestionManager:
    def __init__(self):
        warnings.warn(
            "DataIngestionManager is deprecated and will be removed in version X.X. "
            "Use DataProcessingManager instead.",
            DeprecationWarning,
            stacklevel=2
        )
        self._manager = DataProcessingManager()

    def __getattr__(self, name):
        return getattr(self._manager, name)
```

8. Update the test command to verify changes:
```bash
poetry run pytest tests/ -v
```

This rename provides several benefits:
1. Better alignment between class name and functionality
2. More accurate documentation of the system's capabilities
3. Clearer API for users
4. Consistent terminology throughout the codebase

Would you like me to implement any of these changes, or would you prefer to see a different aspect of the rename process? 