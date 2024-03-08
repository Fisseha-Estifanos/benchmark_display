# Benchmark Display Repository README

## Repository Structure

This repository includes scripts and configurations for benchmarking datasets using the Ragas and Databricks tools. The primary contents are:

- `databricks_benchmarking/`: Folder containing the Databricks benchmarking script and a dataset folder.
- `ragas_benchmarking/`: Folder containing the Ragas benchmarking script and a dataset folder.
- `display.py`: Python script to display benchmark results.
- `display.ipynb`: Jupyter notebook for displaying benchmark results.
- `benchmarking_config.json`: Configuration file for benchmark results display.
- `model_details_config.json`: Configuration file for model details.
- `testing_config.json`: Configuration file for holding parameters for benchmarking purposes.

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
```

## Running Tests

To run benchmarks using Ragas or Databricks tools, follow these steps:

1. Place the desired dataset in the respective folders (databricks_benchmarking/databricks_evaluation_datasets/ or ragas_benchmarking/ragas_evaluation_datasets/).
2. Modify testing_config.json:
   - 2.1 For Ragas: Change the value of the dataset_name under the first item.
   - 2.1. For Databricks: Change the value of the dataset_name under the second item.
3. Execute the benchmarking script:
    - 3.1 For Ragas:
```bash
python ragas_evaluation.py   
```
   - 3.2 For Databricks:
```bash
     python databricks_evaluation.py
```
4. Update benchmarking_config.json with the results from step 3 for display with the results from databricks or ragas.
