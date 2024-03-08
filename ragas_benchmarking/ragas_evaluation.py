import os
import pandas as pd
from dotenv import load_dotenv
load_dotenv()
from datasets import load_dataset
from datasets import Dataset, DatasetDict
from ragas import evaluate
import json
from ragas.metrics import (
    answer_relevancy,
    faithfulness,
    context_recall,
    context_precision,
)

config_file = '../testing_config.json'
with open(config_file, 'r') as file:
        testing_config_data = json.load(file)


# # TODO : uncomment for open AI key
# OPENAI_API_KEY = os.environ['OPEN_API_KEY']
# os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

data_set_split_name = testing_config_data[0]['data_set_split_name']
def create_dataset_dict_from_csv(csv_file):
    df = pd.read_csv(csv_file)
    dataset = Dataset.from_pandas(df)
    dataset_dict = DatasetDict({data_set_split_name: dataset})
    return dataset_dict

# # splits = [350, 400, 450x, 500, 550x, 600, 650x]
# splits = [450]
# for k in splits:
# load the data
# data_set_name = "m-ric_huggingface_doc/our_rag_with_different_splits/latest_dataset_with_retriever_and_llm_cs_"+str(k)+".csv"
# TODO : change your file name here
data_set_name = testing_config_data[0]['data_set_name']

print(f'\n\ndata_set_name: {data_set_name}\n\n')
dataset_dict = create_dataset_dict_from_csv(data_set_name)
print(f'dataset_dict: {dataset_dict}')
print(f'type of dataset_dict: {type(dataset_dict)}')



# # TODO : If your dataset doesn't contain the ground truth, uncomment the code section below
# # TODO : change the dataset_name to the dataset that contains the ground truth 
# # # load the initial dataset with ground truth
# # dataset_name = testing_config_data[0]['ground_truth_containing_data_set_name']
# # print(f"\n\n>- Loading RAG dataset '{dataset_name}' ...")
# # dataset = load_dataset(dataset_name)
# # print(f'dataset:: {dataset}')
# # print(f'type of dataset:: {type(dataset)}')


# # TODO : uncomment the code section below to add the ground truth to your dataset
# # # add the ground truth
# # eval_dataset_with_new_column = dataset_dict[data_set_split_name].add_column('ground_truth', dataset['train']['answer'])
# # dataset_dict[data_set_split_name] = eval_dataset_with_new_column
# # print(f'\n\ndataset_dict::: {dataset_dict}')
# # print(f'type of dataset_dict::: {type(dataset_dict)}')



# convert the list (dataset_dict['contexts']) to a sequence
changed_list_of_contexts = [[item] for item in dataset_dict[data_set_split_name]['contexts']]
list_of_strings = [item[0] for item in changed_list_of_contexts]
def update_contexts(example, index):
    example['contexts'] = [list_of_strings[index]]
    return example
# Use the map function to update the 'contexts' column
dataset_dict[data_set_split_name] = dataset_dict[data_set_split_name].map(update_contexts, with_indices=True,
                                                load_from_cache_file=False)

try:
    # running the evaluation
    result = evaluate(
        dataset_dict[data_set_split_name],
        metrics=[
            context_precision,
            faithfulness,
            answer_relevancy,
            context_recall,
        ],
        raise_exceptions=False
    )
    # print out results
    print(f'result:: {result}')
    df = result.to_pandas()
    print(f'df:: {df}')
except Exception as ex:
    print(f'An error occurred while evaluating the rag:: {ex}')
