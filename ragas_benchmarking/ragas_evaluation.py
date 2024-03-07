import os
import pandas as pd
from dotenv import load_dotenv
load_dotenv()
from datasets import load_dataset
from datasets import Dataset, DatasetDict
from ragas import evaluate
from ragas.metrics import (
    answer_relevancy,
    faithfulness,
    context_recall,
    context_precision,
)

OPENAI_API_KEY = ''
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

def create_dataset_dict_from_csv(csv_file):
    df = pd.read_csv(csv_file)
    dataset = Dataset.from_pandas(df)
    dataset_dict = DatasetDict({'eval': dataset})
    return dataset_dict

# # splits = [350, 400, 450x, 500, 550x, 600, 650x]
# splits = [450]
# for k in splits:
# load the data
# csv_file_path = "m-ric_huggingface_doc/our_rag_with_different_splits/latest_dataset_with_retriever_and_llm_cs_"+str(k)+".csv"
csv_file_path = "../datasets/data_set_name.csv"

print(f'\n\ncsv_file_path: {csv_file_path}\n\n')
dataset_dict = create_dataset_dict_from_csv(csv_file_path)
print(f'dataset_dict: {dataset_dict}')
print(f'type of dataset_dict: {type(dataset_dict)}')



# # load the initial dataset with ground truth
# dataset_name = "m-ric/huggingface_doc_qa_eval"
# print(f"\n\n>- Loading RAG dataset '{dataset_name}' ...")
# dataset = load_dataset(dataset_name)
# print(f'dataset:: {dataset}')
# print(f'type of dataset:: {type(dataset)}')



# # add the ground truth
# eval_dataset_with_new_column = dataset_dict['eval'].add_column('ground_truth', dataset['train']['answer'])
# dataset_dict['eval'] = eval_dataset_with_new_column
# print(f'\n\ndataset_dict::: {dataset_dict}')
# print(f'type of dataset_dict::: {type(dataset_dict)}')



# convert the list (dataset_dict['contexts']) to a sequence
changed_list_of_contexts = [[item] for item in dataset_dict['eval']['contexts']]
list_of_strings = [item[0] for item in changed_list_of_contexts]
def update_contexts(example, index):
    example['contexts'] = [list_of_strings[index]]
    return example
# Use the map function to update the 'contexts' column
dataset_dict['eval'] = dataset_dict['eval'].map(update_contexts, with_indices=True,
                                                load_from_cache_file=False)

try:
    # running the evaluation
    result = evaluate(
        dataset_dict['eval'],
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
