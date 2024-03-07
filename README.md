# benchmark_display

step 1 : environmental configurations
git clone repo
cd repo
python3 -m venv venv
source venv/bin/activate

step 2 : Install modules
pip3 install --r requirements.txt



Step 3 : load in the datasets
The datasets in the datasets folder need to be in the following format
For Ragas
["question", "ground_truth", "answer", "contexts"]


For DataBricks
["question", "answer", "context"]pyth 


Step 4 : Benchmark the datasets



Step 4 : Display the benchmarking results
Display notebook
run the notebook


Display benchmarking results in stream lit
streamlit display.py