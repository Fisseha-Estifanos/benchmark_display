# Benchmark Display Repository README

## Repository Structure

This repository includes scripts and configurations for benchmarking datasets using the Ragas and Databricks tools. The primary contents are:

- `databricks_benchmarker/`: Folder containing the Databricks benchmarking script and a dataset folder.
- `ragas_benchmarker/`: Folder containing the Ragas benchmarking script and a dataset folder.
- `display.py`: Python script to display benchmark results.
- `display.ipynb`: Jupyter notebook for displaying benchmark results.
- `benchmarking_config.json`: Configuration file for benchmark results display.
- `model_details_config.json`: Configuration file for model details.
- `testing_config.json`: Configuration file for test parameters.

## Setting Up the Repository Locally

Follow these steps to recreate the repository on your local machine. Ensure you have Bash shell and Python 3.10 installed.

```bash
# 1. Clone the repository
git clone https://github.com/Fisseha-Estifanos/benchmark_display.git

# 2. Navigate to the repository directory
cd benchmark_display/

# 3. Create a virtual environment
python3.10 -m venv venv

# 4. Activate the virtual environment
source venv/bin/activate

# 5. Upgrade pip
pip install --upgrade pip

# 6. Install required dependencies
pip install -r requirements.txt
