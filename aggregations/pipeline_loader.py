"""
Pipeline loader utilities for MongoDB aggregation pipelines.

This module provides utilities for loading MongoDB aggregation pipelines from JSON files,
making them reusable across different tools in the project.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, List

# Setup logger for this module
logger = logging.getLogger(__name__)


def load_aggregation_pipeline(pipeline_file: str, base_path: str = None) -> List[Dict[str, Any]]:
    """
    Load MongoDB aggregation pipeline from JSON file

    Args:
        pipeline_file: Path to the JSON file containing the aggregation pipeline
        base_path: Base path to resolve relative paths against. If None, uses the 
                   aggregations directory as the base path.

    Returns:
        List of aggregation stages

    Raises:
        ValueError: If the pipeline file cannot be loaded or parsed
    """
    pipeline_path = Path(pipeline_file)

    # If path is not absolute, make it relative to the aggregations directory
    if not pipeline_path.is_absolute():
        if base_path:
            pipeline_path = Path(base_path) / pipeline_file
        else:
            # Default to the aggregations directory
            aggregations_dir = Path(__file__).parent
            pipeline_path = aggregations_dir / pipeline_file

    try:
        with open(pipeline_path, 'r', encoding='utf-8') as f:
            pipeline = json.load(f)
        logger.debug("Loaded aggregation pipeline from %s", pipeline_path)
        return pipeline
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logger.error("Error loading aggregation pipeline from %s: %s", pipeline_path, e)
        raise ValueError(f"Could not load aggregation pipeline from {pipeline_path}: {e}") from e
