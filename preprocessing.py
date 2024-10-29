#!/usr/bin/env python
# coding: utf-8
from utils import read_jsonl_file, write_jsonl_file
from utils import build_zero_shot_prompt
from utils import get_embedding
from utils import get_response, parse_answer, validate_response
from utils import system_prompt
from tqdm import tqdm
import os
import yaml
import sys

class PreProcessor:
    def __init__(self, configs):
        """
        Constructor for PreProcessor.

        Parameters
        ----------
        configs : dict
            A dictionary containing configuration parameters.
            The dictionary should contain the following keys:
            - train_data_file: the file name of the training data file
            - result_folder: the name of the folder where the preprocessed data will be saved
            - embedding_model: the name of the embedding model to use (e.g. "all-minilm:22m")

        Returns
        -------
        None
        """
        self.data_fn = configs["train_data_file"]
        self.result_folder = configs["result_folder"]
        if not os.path.isdir(self.result_folder):
            os.mkdir(self.result_folder)
        self.cot_json_fn = os.path.join(self.result_folder, "cot_responses_medqa_train_set.jsonl")
        self.cot_json_fn_filtered_with_embeddings = os.path.join(self.result_folder, "cot_responses_medqa_train_set_filtered_with_embeddings.jsonl")
        self.embedding_model = configs["embedding_model"]

    def preprocess(self):
        """
        Preprocess the data by first generating chain of thought (CoT) responses for each data sample,
        then filtering out the samples with incorrect predicted answers and embedding the questions.

        If the CoT responses file does not exist, generate the CoT responses and save them to a file.
        If the file with filtered and embedded questions does not exist, filter out the samples with incorrect
        predicted answers and embed the questions, then save the results to a file.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        if not os.path.exists(self.cot_json_fn):
            data = self.load_data()
            qa_dict = self.generate_cots(data)
            self.filtered_qa_dict = self.filter_questions(qa_dict)
            self.embed_questions()
        elif not os.path.exists(self.cot_json_fn_filtered_with_embeddings):
            qa_dict = read_jsonl_file(self.cot_json_fn)
            self.filtered_qa_dict = self.filter_questions(qa_dict)
            self.embed_questions()

    # Load data with necessary filtering
    def load_data(self):
        """
        Load data with necessary filtering

        Parameters
        ----------
        None

        Returns
        -------
        list
            List of dictionaries, where each dictionary contains the data for a single sample.
        """
        data = read_jsonl_file(self.data_fn)

        # Limit the data to 2000 samples
        # TODO: Sampling with a strategy to cover diverse distributions
        if len(data) > 2000:
            data = data[:2000]
            
        return data

    # Generate chain of thoughts and answers
    def generate_cots(self, data):
        """
        Generate chain of thought (CoT) responses for each data sample and save the results.

        This method processes each item in the provided data by constructing a prompt and retrieving a response
        from a specified model. The response is then parsed to extract the chain of thought and the predicted answer.
        The processed results are saved to a file, and a list of dictionaries with the relevant information is returned.

        Parameters
        ----------
        data : list of dict
            A list of dictionaries, where each dictionary contains information about a sample, 
            including 'question', 'answer', and 'options'.

        Returns
        -------
        list of dict
            A list of dictionaries, where each dictionary contains 'idx', 'question', 'answer', 
            'options', 'cot' (chain of thought), and 'pred_ans' (predicted answer).
        """
        # Create a folder for cot responses
        cot_folder = os.path.join(self.result_folder, "cot_responses")
        if not os.path.isdir(cot_folder):
            os.mkdir(cot_folder)
        qa_dict = []
        ab_normal_cot = 0
        for idx, item in enumerate(tqdm(data)):    
            prompt = build_zero_shot_prompt(system_prompt, item)
            try:
                response = get_response(prompt, model_name="llama3.2:1b", max_tokens=2000)
                with open(os.path.join(cot_folder, str(idx) + ".txt"), "w", encoding="utf-8") as f:
                    f.write(response)

                # Convert the response to a dictionary
                # In case the response is not valid
                if not validate_response(response):
                    ab_normal_cot += 1
                    continue

                # Parse the response
                # TODO: add answer_idx to avoid unnecessary mapping between answer and answer_idx
                cot, pred_ans = parse_answer(response)
                dict_elem = {}
                dict_elem["idx"] = idx
                dict_elem["question"] = item["question"]
                dict_elem["answer"] = item["answer"]
                dict_elem["options"] = item["options"]
                dict_elem["cot"] = cot
                dict_elem["pred_ans"] = pred_ans
                qa_dict.append(dict_elem)

            except Exception as e :
                print(str(e))
        
        print(">>> No. of abnormal cot responses: ", ab_normal_cot)
        
        write_jsonl_file(self.cot_json_fn, qa_dict)

        return qa_dict  
        
    # Filter questions whose predicted answer does not match the actual answer
    def filter_questions(self, qa_dict):
        """
        Filter questions whose predicted answer does not match the actual answer.

        Parameters
        ----------
        qa_dict : list
            A list of dictionaries, where each dictionary contains information about a sample, 
            including 'idx', 'question', 'answer', 'options', 'cot' (chain of thought), and 'pred_ans' (predicted answer).

        Returns
        -------
        filtered_qa_dict : list
            A filtered list of dictionaries, where each dictionary contains the same information as the input,
            but only for the samples whose predicted answer matches the actual answer.
        """
        filtered_qa_dict = []
        for item in tqdm(qa_dict):
            # TODO: Change to use answer_idx, no need to do this mapping
            pred_ans = item["options"][item["pred_ans"]]
            if pred_ans == item["answer"]:
                filtered_qa_dict.append(item)

        print(">>> No. of Raw Responses: ", len(qa_dict))
        print(">>> No. of Good Responses: ", len(filtered_qa_dict))

        return filtered_qa_dict


    # Embed all questions
    def embed_questions(self):
        """
        Embed all questions in the filtered data with the specified embedding model.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        for item in tqdm(self.filtered_qa_dict):
            item["embedding"] = get_embedding(item["question"], model=self.embedding_model)

            # TODO: Change to use answer_idx, no need to do this mapping
            inv_options_map = {v:k for k,v in item["options"].items()}
            item["answer_idx"] = inv_options_map[item["answer"]]

        # Write the final processed data to a jsonl file
        write_jsonl_file(self.cot_json_fn_filtered_with_embeddings, self.filtered_qa_dict)


# Main function
if __name__ == "__main__":

    # Load the YAML configuration file
    def load_config(file_path):
        with open(file_path, "r") as file:
            config = yaml.safe_load(file)
        return config
    
    if len(sys.argv) > 1:
        config_fn = sys.argv[1]
    else: 
        config_fn = "configs/default.yaml"
    configs = load_config(config_fn)
    print(configs)

    preproc = PreProcessor(configs)
    preproc.preprocess()