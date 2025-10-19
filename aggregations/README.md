# Aggregations Module

This module provides utilities for working with MongoDB aggregation pipelines in the ask-mongo-jira project.

## Usage

### Loading Aggregation Pipelines

Use the `load_aggregation_pipeline` function to load MongoDB aggregation pipeline definitions from JSON files:

```python
from aggregations import load_aggregation_pipeline

# Load a pipeline from the aggregations directory
pipeline = load_aggregation_pipeline("analyze_code_changes_aggregation.json")

# Load a pipeline with a custom base path
pipeline = load_aggregation_pipeline("my_pipeline.json", base_path="/path/to/pipelines")

# Load a pipeline with an absolute path
pipeline = load_aggregation_pipeline("/absolute/path/to/pipeline.json")
```

### Function Reference

#### `load_aggregation_pipeline(pipeline_file: str, base_path: str = None) -> List[Dict[str, Any]]`

Loads a MongoDB aggregation pipeline from a JSON file.

**Parameters:**
- `pipeline_file`: Path to the JSON file containing the aggregation pipeline
- `base_path`: Optional base path to resolve relative paths against. If None, uses the aggregations directory as the base path.

**Returns:**
- List of aggregation stages (dictionaries)

**Raises:**
- `ValueError`: If the pipeline file cannot be loaded or parsed

## Pipeline Files

Pipeline JSON files should be stored in the `aggregations/` directory and contain valid MongoDB aggregation pipeline stages as a JSON array.

Example:
```json
[
    {"$match": {"epic": "PLACEHOLDER"}},
    {"$lookup": {...}},
    {"$unwind": {...}}
]
```