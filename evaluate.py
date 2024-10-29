#!/usr/bin/env python
# coding: utf-8
from utils import read_jsonl_file, write_jsonl_file
from utils import build_zero_shot_prompt, build_few_shot_prompt
from utils import get_embedding
from utils import get_response, parse_answer, validate_response
from utils import find_mode_string_list, extract_ans_option
from utils import system_prompt, system_zero_shot_prompt
from tqdm import tqdm
import os
import yaml
import sys
import numpy as np
from sklearn.neighbors import NearestNeighbors
import random


class Inferencer():
    def __init__(self, configs):
        """
        Constructor for Inferencer.

        Parameters
        ----------
        configs : dict
            A dictionary containing configuration parameters.
            The dictionary should contain the following keys:
            - test_data_file: the file name of the test data file
            - result_folder: the name of the folder where the preprocessed data was saved
            - embedding_model: the name of the embedding model to use (e.g. "all-minilm:22m")
            - eval_types: a list of strings, each of which can be either "kNN_few_shot_cot", "few_shot_cot", "few_shot", or "zero_shot"

        Returns
        -------
        None
        """
        self.test_data_fn = configs["test_data_file"]
        self.result_folder = configs["result_folder"]
        assert os.path.exists(self.result_folder), "!!! Result folder does not exist, Please run preprocessing.py first !!!"
        self.cot_json_fn_filtered_with_embeddings = os.path.join(self.result_folder, "cot_responses_medqa_train_set_filtered_with_embeddings.jsonl")
        self.final_processed_test_set_responses_medprompt = os.path.join(self.result_folder,"final_processed_test_set_responses_medprompt.jsonl")
        self.test_set_responses_kNN_few_shot_cot_fn = os.path.join(self.result_folder,"test_set_responses_kNN_few_shot_cot.jsonl")
        self.test_set_responses_few_shot_cot_fn = os.path.join(self.result_folder,"test_set_responses_few_shot_cot.jsonl")
        self.test_set_responses_few_shot_fn = os.path.join(self.result_folder,"test_set_responses_few_shot.jsonl")
        self.test_set_responses_zero_shot_fn = os.path.join(self.result_folder,"test_set_responses_zero_shot.jsonl")
        self.embedding_model = configs["embedding_model"]
        self.eval_types = configs["eval_types"]

    def infer(self):
        """
        Perform inference on the test samples.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.test_samples = self.load_data()
        self.n_test_samples = len(self.test_samples)
        self.filtered_qa_dict = read_jsonl_file(self.cot_json_fn_filtered_with_embeddings)
        print(">>> Loaded {} examples".format(len(self.filtered_qa_dict)))
        self.random_few_shot_examples = random.sample(self.filtered_qa_dict, 5)
        self.intialize_knn()
        for i in range(len(self.eval_types)):
            curr_eval_type = self.eval_types[i]
            print(">>>>>> Current eval type: ", curr_eval_type)
            if curr_eval_type == "kNN_few_shot_cot_ensemble":
                self.kNN_few_shot_cot_ensemble()
                self.post_process_ensemble()
            elif curr_eval_type == "kNN_few_shot_cot":
                self.kNN_few_shot_cot()
            elif curr_eval_type == "few_shot_cot":
                self.few_shot_cot()
            elif curr_eval_type == "few_shot":
                self.few_shot()
            elif curr_eval_type == "zero_shot":
                self.zero_shot()
            else:
                raise ValueError("Unknown eval type {}".format(curr_eval_type))
        
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
        data = read_jsonl_file(self.test_data_fn)
        print(">>> Loaded {} test samples".format(len(data)))

        return data
    
    def intialize_knn(self):
        
        """
        Initializes and trains a k-Nearest Neighbors (kNN) model using the precomputed embeddings.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        # Extract embeddings and keep track of indices
        embeddings = np.array([d["embedding"] for d in self.filtered_qa_dict])

        # Train KNN model
        self.knn = NearestNeighbors(n_neighbors=3, algorithm='auto', metric='cosine', n_jobs=-1).fit(embeddings)

    def shuffle_option_labels(self, answer_options):
        """
        Shuffle the labels of the answer options.

        Parameters
        ----------
        answer_options : dict
            A dictionary containing the answer options to be shuffled.

        Returns
        -------
        dict
            A dictionary with shuffled answer options and corresponding labels.
        """
        options = list(answer_options.values())
        random.shuffle(options)
        labels = [chr(i) for i in range(ord('A'), ord('A') + len(options))]
        shuffled_options_dict = {label: option for label, option in zip(labels, options)}
        
        return shuffled_options_dict

    def kNN_few_shot_cot_ensemble(self):
        """
        Evaluate the model using the k-Nearest Neighbors (kNN) model as a way to generate few-shot prompts with shuffled options.
        
        Parameters
        ----------
        None
        
        Returns
        -------
        None
        """
        for question in tqdm(self.test_samples, colour ="green"):
            question_variants = []
            prompt_variants = []
            cot_responses = []
            question_embedding = get_embedding(question["question"], model=self.embedding_model)
            distances, top_k_indices = self.knn.kneighbors([question_embedding], n_neighbors=5)
            top_k_dicts = [self.filtered_qa_dict[i] for i in top_k_indices[0]]
            question["outputs"] = []
    
            # Generate five prompts with shuffled options
            for idx in range(5):
                question_copy = question.copy()
                shuffled_options = self.shuffle_option_labels(question["options"])
                inv_map = {v:k for k,v in shuffled_options.items()}
                
                question_copy["options"] = shuffled_options
                question_copy["answer_idx"] = inv_map[question_copy["answer"]]
                question_variants.append(question_copy)
                prompt = build_few_shot_prompt(system_prompt,  question_copy, top_k_dicts)
                prompt_variants.append(prompt)
            
            # Get responses for the five prompts
            for prompt in tqdm(prompt_variants):
                response = get_response(prompt, model_name="llama3.2:1bfy", max_tokens=1000)
                cot_responses.append(response)
            
            for question_sample, answer in zip(question_variants, cot_responses):
                if validate_response(answer):
                    cot, pred_ans = parse_answer(answer)
                    
                else:
                    cot = ""
                    pred_ans = ""
                
                # TODO: Change to use answer_idx, no need to do the mapping between answer and answer_idx
                question["outputs"].append({"question": question_sample["question"], "options": question_sample["options"], "cot": cot, "pred_ans": question_sample["options"].get(pred_ans, "")})

    def post_process_ensemble(self):
        """
        Post process the output of the kNN few shot prompt ensemble model.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        ctr = 0 
        for idx,item in enumerate(self.test_samples):
            pred_ans = [x["pred_ans"] for x in item["outputs"]]
            freq_ans = find_mode_string_list(pred_ans)
            
            if len(freq_ans) > 1:
                final_prediction = ""
            
            else:
                final_prediction = freq_ans[0]
            
            item["final_prediction"] = final_prediction

            # TODO: Change to use answer_idx, no need to do the mapping between answer and answer_idx
            if final_prediction == item["answer"]:
                ctr += 1

        print(">>> Total true predictions: ", ctr)
        print(">>> Total test samples: ", self.n_test_samples)
        print(">>> Accuracy: ", ctr/self.n_test_samples)

        write_jsonl_file(self.final_processed_test_set_responses_medprompt, self.test_samples)

    def kNN_few_shot_cot(self):
        """
        Perform k-Nearest Neighbors (kNN) based few-shot chain-of-thought (CoT) inference.

        This method processes each test sample by finding its k nearest neighbors using precomputed
        embeddings. It then generates a prompt using these neighbors and a system prompt to obtain
        a response from a language model. The response is parsed to extract the predicted answer and
        chain of thought. The predicted answer is compared against the true answer index, and the
        accuracy of predictions is calculated.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        ctr = 0
        for question in tqdm(self.test_samples, colour ="green"):
            question_embedding = get_embedding(question["question"], model=self.embedding_model)
            distances, top_k_indices = self.knn.kneighbors([question_embedding], n_neighbors=5)
            top_k_dicts = [self.filtered_qa_dict[i] for i in top_k_indices[0]]
    
            # Get response
            prompt = build_few_shot_prompt(system_prompt,  question, top_k_dicts)
            response = get_response(prompt, model_name="llama3.2:1bfy", max_tokens=1000)
   
            if validate_response(response):
                cot, pred_ans = parse_answer(response)
                
            else:
                cot = ""
                pred_ans = ""
            question["cot"] = cot
            question["pred_ans"] = pred_ans

            if pred_ans == question["answer_idx"]:
                ctr += 1

        print(">>> Total true predictions: ", ctr)
        print(">>> Total test samples: ", self.n_test_samples)
        print(">>> Accuracy: ", ctr/self.n_test_samples)
        
        write_jsonl_file(self.test_set_responses_kNN_few_shot_cot_fn, self.test_samples)

    def few_shot_cot(self):
        """
        Perform few-shot chain-of-thought (CoT) inference.

        This method processes each test sample by generating a prompt using random few-shot examples
        and a system prompt to obtain a response from a language model. The response is parsed to
        extract the predicted answer and chain of thought. The predicted answer is compared against
        the true answer index, and the accuracy of predictions is calculated.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        ctr = 0
        for item in tqdm(self.test_samples):
            messages = build_few_shot_prompt(system_prompt, item, self.random_few_shot_examples, include_cot=True)
            response = get_response(messages, "llama3.2:1bfy", max_tokens=1000)

            if validate_response(response):
                cot, pred_ans = parse_answer(response)
            else:
                cot = ""
                pred_ans = ""

            item["pred_ans"] = pred_ans
            item["cot"] = cot

            if pred_ans == item["answer_idx"]:
                ctr += 1

        print(">>> Total true predictions: ", ctr)
        print(">>> Total test samples: ", self.n_test_samples)
        print(">>> Accuracy: ", ctr/self.n_test_samples)
        
        write_jsonl_file(self.test_set_responses_few_shot_cot_fn, self.test_samples)

    def few_shot(self):
        """
        Perform few-shot inference without chain-of-thought (CoT) explanations.

        This method processes each test sample by generating a prompt using random few-shot examples
        and a system prompt to obtain a response from a language model. The response is parsed to
        extract the predicted answer. The predicted answer is compared against the true answer index,
        and the accuracy of predictions is calculated.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        ctr = 0
        for item in tqdm(self.test_samples):
            messages = build_few_shot_prompt(system_zero_shot_prompt, item, self.random_few_shot_examples, include_cot=False)
            response = get_response(messages, "llama3.2:1bfy", max_tokens=1000)
            pred_ans = extract_ans_option(response.split("\n")[-1])
            item["pred_ans"] = pred_ans

            if pred_ans == item["answer_idx"]:
                ctr += 1

        print(">>> Total true predictions: ", ctr)
        print(">>> Total test samples: ", self.n_test_samples)
        print(">>> Accuracy: ", ctr/self.n_test_samples)
        
        write_jsonl_file(self.test_set_responses_few_shot_fn, self.test_samples)


    def zero_shot(self):
        """
        Perform zero-shot inference without few-shot examples or chain-of-thought (CoT) explanations.

        This method processes each test sample by generating a prompt using a system prompt to obtain
        a response from a language model. The response is parsed to extract the predicted answer. The
        predicted answer is compared against the true answer index, and the accuracy of predictions is
        calculated.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        ctr = 0
        for item in tqdm(self.test_samples):
            messages = build_zero_shot_prompt(system_zero_shot_prompt, item)
            response = get_response(messages, "llama3.2:1bfy", max_tokens=1000)
            pred_ans = extract_ans_option(response.split("\n")[-1])
            item["pred_ans"] = pred_ans

            if pred_ans == item["answer_idx"]:
                ctr += 1

        print(">>> Total true predictions: ", ctr)
        print(">>> Total test samples: ", self.n_test_samples)
        print(">>> Accuracy: ", ctr/self.n_test_samples)
        
        write_jsonl_file(self.test_set_responses_zero_shot_fn, self.test_samples)


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

    inferer = Inferencer(configs)
    inferer.infer()