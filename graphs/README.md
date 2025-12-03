# MongoDB Analysis Graphs

This folder contains Python scripts and generated outputs for analyzing MongoDB JIRA epic data.

## Scripts

- **`generate_graphs.py`** - Main script to generate both scatter plot and histogram with saved PNG files
- **`generate_graph_data.py`** - Generate CSV data files for further analysis  
- **`interactive_graph.py`** - Interactive scatter plot with hover tooltips
- **`__init__.py`** - Python package marker

## Usage

From the main project directory:

```bash
# Generate all graphs and data
python graphs/generate_graphs.py

# Generate CSV data only
python graphs/generate_graph_data.py

# Interactive plot with tooltips
python graphs/interactive_graph.py
```

## Outputs

Generated files are saved in this `graphs/` directory:

- **PNG Files**: Static images for reports and presentations
- **CSV Files**: Raw data for further analysis or import into other tools
- **Interactive Windows**: Live matplotlib plots with hover functionality

## Requirements

- MongoDB connection (MONGODB_URL environment variable)
- Python packages: matplotlib, numpy, motor, pymongo, dotenv