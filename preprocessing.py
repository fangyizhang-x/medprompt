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
            - embedding_model: the name of the embedding model to use (e.g. "text-embedding-ada-002")

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
        This function is the main entry point for preprocessing data. It loads data and 
        generates CoT responses for all samples in the data. It also filters out samples 
        whose predicted answer does not match the actual answer and embeds the questions in 
        the filtered data. The preprocessed data is saved as a jsonl file in the specified 
        folder. The function will not re-run the entire preprocessing process if the output 
        file already exists. It will simply load the preprocessed data from the output file.
        """
        if not os.path.exists(self.cot_json_fn):
            data = self.load_data()
            qa_dict = self.generate_cots(data)
        else:
            if not os.path.exists(self.cot_json_fn_filtered_with_embeddings):
                qa_dict = read_jsonl_file(self.cot_json_fn)
                self.filtered_qa_dict = self.filter_questions(qa_dict)
                self.embed_questions()

    # Load data with necessary filtering
    def load_data(self):
        """
        Loads data from the specified file and performs necessary filtering on the data.

        :return: A list of dictionaries, where each dictionary represents a sample in the data.
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
        Generates chain of thought (CoT) responses for each data sample and saves the results.

        This function processes each item in the given data by building a prompt and obtaining a response
        from a specified model. The response is parsed to extract the chain of thought and predicted answer.
        The results are saved to a file, and a list of dictionaries with the processed information is returned.

        :param data: A list of dictionaries, where each dictionary contains information about a sample, 
                    including 'question', 'answer', and 'options'.
        :return: A list of dictionaries, where each dictionary contains the 'idx', 'question', 'answer', 
                'options', 'cot' (chain of thought), and 'pred_ans' (predicted answer).
        """
        # Create a folder for cot responses
        cot_folder = os.path.join(self.result_folder, "cot_responses")
        if not os.path.isdir(cot_folder):
            os.mkdir(cot_folder)
        qa_dict = []
        for idx, item in enumerate(tqdm(data)):    
            prompt = build_zero_shot_prompt(system_prompt, item)
            try:
                response = get_response(prompt, model_name="llama3.2:1b", max_tokens=500)
                with open(os.path.join(cot_folder, str(idx) + ".txt"), "w", encoding="utf-8") as f:
                    f.write(response)

                # Convert the response to a dictionary
                # In case the response is not valid
                if not validate_response(response):
                    continue

                # Parse the response
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
        
        write_jsonl_file(self.cot_json_fn, qa_dict)

        return qa_dict  
        
    # Filter questions whose predicted answer does not match the actual answer
    def filter_questions(self, qa_dict):
        """
        Filter questions whose predicted answer does not match the actual answer.

        Args:
            qa_dict (list): A list of dictionaries, where each dictionary represents a sample in the data.
                Each dictionary should contain the keys "idx", "question", "answer", "options", "cot", and "pred_ans".

        Returns:
            filtered_qa_dict (list): A list of dictionaries, where each dictionary represents a sample in the filtered data.
        """
        filtered_qa_dict = []
        for item in tqdm(qa_dict):
            pred_ans = item["options"][item["pred_ans"]]
            if pred_ans == item["answer"]:
                filtered_qa_dict.append(item)

        print(">>> No. of Raw Responses: ", len(qa_dict))
        print(">>> No. of Good Responses: ", len(filtered_qa_dict))

        return filtered_qa_dict


    # Embed all questions
    def embed_questions(self):
        """
        Embeds all questions in the filtered data using the specified embedding model.

        Modifies each item in the filtered data by adding an "embedding" key with the embedding of the question.
        Also adds an "answer_idx" key with the index of the correct answer in the options list.

        :return: None
        """
        for item in tqdm(self.filtered_qa_dict):
            item["embedding"] = get_embedding(item["question"], model=self.embedding_model)
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