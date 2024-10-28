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
from multiprocessing import Process


class Inferencer():
    def __init__(self, configs):
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
        self.test_samples = self.load_data()
        self.n_test_samples = len(self.test_samples)
        self.filtered_qa_dict = read_jsonl_file(self.cot_json_fn_filtered_with_embeddings)
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
        data = read_jsonl_file(self.test_data_fn)
        # data = data[:10]
        print(">>> Loaded {} test samples".format(len(data)))

        return data
    
    def intialize_knn(self):
        # Extract embeddings and keep track of indices
        embeddings = np.array([d["embedding"] for d in self.filtered_qa_dict])
        # indices = list(range(len(self.filtered_qa_dict))) # No need

        # Train KNN model
        self.knn = NearestNeighbors(n_neighbors=3, algorithm='auto', metric='cosine', n_jobs=-1).fit(embeddings)

    def shuffle_option_labels(self, answer_options):
        """
        Shuffles the options of the question.
        
        Parameters:
        answer_options (dict): A dictionary with the options.

        Returns:
        dict: A new dictionary with the shuffled options.
        """
        options = list(answer_options.values())
        random.shuffle(options)
        labels = [chr(i) for i in range(ord('A'), ord('A') + len(options))]
        shuffled_options_dict = {label: option for label, option in zip(labels, options)}
        
        return shuffled_options_dict

    def kNN_few_shot_cot_ensemble(self):
        for question in tqdm(self.test_samples, colour ="green"):
            question_variants = []
            prompt_variants = []
            cot_responses = []
            question_embedding = get_embedding(question["question"])
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
                response = get_response(prompt, model_name="llama3.2:1b", max_tokens=500)
                cot_responses.append(response)
            
            for question_sample, answer in zip(question_variants, cot_responses):
                if validate_response(answer):
                    cot, pred_ans = parse_answer(answer)
                    
                else:
                    cot = ""
                    pred_ans = ""
                        
                question["outputs"].append({"question": question_sample["question"], "options": question_sample["options"], "cot": cot, "pred_ans": question_sample["options"].get(pred_ans, "")})

    def post_process_ensemble(self):
        ctr = 0 
        for idx,item in enumerate(self.test_samples):
            pred_ans = [x["pred_ans"] for x in item["outputs"]]
            freq_ans = find_mode_string_list(pred_ans)
            
            if len(freq_ans) > 1:
                final_prediction = ""
            
            else:
                final_prediction = freq_ans[0]
            
            item["final_prediction"] = final_prediction

            if final_prediction == item["answer"]:
                ctr += 1

        print(">>> Total true predictions: ", ctr)
        print(">>> Total test samples: ", self.n_test_samples)
        print(">>> Accuracy: ", ctr/self.n_test_samples)

        write_jsonl_file(self.final_processed_test_set_responses_medprompt, self.test_samples)

    def kNN_few_shot_cot(self):
        ctr = 0
        for question in tqdm(self.test_samples, colour ="green"):
            question_embedding = get_embedding(question["question"])
            distances, top_k_indices = self.knn.kneighbors([question_embedding], n_neighbors=5)
            top_k_dicts = [self.filtered_qa_dict[i] for i in top_k_indices[0]]
    
            # Get response
            prompt = build_few_shot_prompt(system_prompt,  question, top_k_dicts)
            response = get_response(prompt, model_name="llama3.2:1b", max_tokens=500)
   
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
        ctr = 0
        for item in tqdm(self.test_samples):
            messages = build_few_shot_prompt(system_prompt, item, self.random_few_shot_examples, include_cot=True)
            response = get_response(messages, "llama3.2:1b", max_tokens=500)

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
        ctr = 0
        for item in tqdm(self.test_samples):
            messages = build_few_shot_prompt(system_zero_shot_prompt, item, self.random_few_shot_examples, include_cot=False)
            response = get_response(messages, "llama3.2:1b", max_tokens=500)
            pred_ans = extract_ans_option(response.split("\n")[-1])
            item["pred_ans"] = pred_ans

            if pred_ans == item["answer_idx"]:
                ctr += 1

        print(">>> Total true predictions: ", ctr)
        print(">>> Total test samples: ", self.n_test_samples)
        print(">>> Accuracy: ", ctr/self.n_test_samples)
        
        write_jsonl_file(self.test_set_responses_few_shot_fn, self.test_samples)


    def zero_shot(self):
        ctr = 0
        for item in tqdm(self.test_samples):
            messages = build_zero_shot_prompt(system_zero_shot_prompt, item)
            response = get_response(messages, "llama3.2:1b", max_tokens=500)
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

    # processes = []
    # # Create and start multiple processes
    # for i in range(10):  # Change the range to create more or fewer processes
    #     p = Process(target=inferer.infer())
    #     processes.append(p)
    #     p.start()

    # # Wait for all processes to complete
    # for p in processes:
    #     p.join()

    # print('All workers have finished.')
    