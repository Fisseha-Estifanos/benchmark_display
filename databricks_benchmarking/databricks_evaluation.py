import os
import json
from dotenv import load_dotenv
load_dotenv()

# # TODO : uncomment for open AI key
# OPENAI_API_KEY = os.environ['OPEN_API_KEY']
# os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# Setup the evaluator prompts
from databricks.labs.doc_qa.llm_utils import PromptTemplate
import pandas as pd
from databricks.labs.doc_qa.evaluators.templated_evaluator import OpenAIEvaluator, AnthropicEvaluator, ParameterDef, NoRetryPolicy, RetryPolicy
from databricks.labs.doc_qa.variables.doc_qa_template_variables import get_openai_grading_template_and_function

import logging
logging.basicConfig(level=logging.INFO)


config_file = '../testing_config.json'
with open(config_file, 'r') as file:
        testing_config_data = json.load(file)


retry_policy = RetryPolicy(max_retry_on_invalid_result=3,
                           max_retry_on_exception=3)
catch_error = True

openai_grading_prompt, openai_grading_function = get_openai_grading_template_and_function(
        scale=1, level_of_details=2)

## Load answer sheet (csv)

# splits = [200, 250, 300, 350, 400, 450, 500, 550, 600, 650]
# for k in splits:
# data_set_name="m-ric_huggingface_doc/our_rag_on_different_splits/latest_dataset_with_retriever_and_llm_cs_"+str(k)+".csv"
# TODO : change your file name here
data_set_name = testing_config_data[1]['data_set_name']
target_df = pd.read_csv(data_set_name)

# TODO : change the run_the_benchmarking_test_on value to test smaller number of rows
run_the_benchmarking_test_on=len(target_df)
print(target_df)
print(f'data_set_name: {data_set_name}')
print(f'total length: {len(target_df)}')

## Evaluate and show results:
data_bricks_evaluation_llm = testing_config_data[1]['data_bricks_evaluation_llm']
temp = testing_config_data[1]['temperature']
input_columns = testing_config_data[1]['input_columns']

openai_gpt_4_evaluator = OpenAIEvaluator(model=data_bricks_evaluation_llm, temperature=temp, 
    grading_prompt_tempate=openai_grading_prompt, 
    input_columns=input_columns,
    openai_function=openai_grading_function,
    retry_policy=retry_policy)
eval_result = openai_gpt_4_evaluator.run_eval(dataset_df=target_df[:run_the_benchmarking_test_on],
                                                concurrency=20,
                                                catch_error=catch_error)
result_df = eval_result.to_dataframe()
print(eval_result.summary()+'\n\n\n')
result_df
