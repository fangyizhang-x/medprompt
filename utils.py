#!/usr/bin/env python
# coding: utf-8

import json
import re
from openai import OpenAI
import re
from collections import Counter

client = OpenAI(
    base_url='http://localhost:11434/v1/',

    # required but ignored
    api_key='ollama',
)

# # Version0.1
# system_prompt = """You are an expert medical professional. You are provided with a medical question with multiple answer choices.
# Your goal is to think through the question carefully and explain your reasoning step by step before selecting the final answer.
# Respond only with the reasoning steps and answer as specified below.
# Please make sure your final answer is a single letter indicating the answer choice you selected.
# Below is the format for each question and answer:

# Input:
# ## Question: {{question}}
# {{answer_choices}}

# Output:
# ## Answer
# (model generated chain of thought explanation)
# Therefore, the answer is [final model answer (e.g. A,B,C,D)]"""

# Version0.2
system_prompt = """
You are a highly knowledgeable medical professional. For each question provided, carefully evaluate the answer choices and explain your reasoning step by step before reaching a final answer.

Instructions:
- Think through the question and outline your reasoning in a clear, logical sequence.
- Conclude with a single letter indicating your chosen answer by explicitly saying: "Therefore, the answer is [final choice: A, B, C, or D]".

Response format:

Input:
## Question: {{question}}
{{answer_choices}}

Output:
## Answer
[Step-by-step reasoning goes here]
Therefore, the answer is [final choice: A, B, C, or D].
"""

system_zero_shot_prompt = """You are an expert medical professional. You are provided with a medical question with multiple answer choices.
Your goal is to think through the question carefully and respond directly with the answer option.
Please make sure your final answer is a single letter indicating the answer choice you selected.
Below is the format for each question and answer:

Input:
## Question: {{question}}
{{answer_choices}}

Output:
## Answer
Therefore, the answer is [final model answer (e.g. A,B,C,D)]"""

def write_jsonl_file(file_path, dict_list):
    """
    Write a list of dictionaries to a JSON Lines file.

    Parameters
    ----------
    file_path : str
        The path to the file where the data will be written.
    dict_list : list
        A list of dictionaries to write to the file.

    Returns
    -------
    None
    """
    with open(file_path, 'w') as file:
        for dictionary in dict_list:
            json_line = json.dumps(dictionary)
            file.write(json_line + '\n')

def read_jsonl_file(file_path):
    """
    Reads a JSON Lines file and returns a list of dictionaries.

    Parameters
    ----------
    file_path : str
        The path to the JSON Lines file to be read.

    Returns
    -------
    list of dict
        A list where each element is a dictionary representing a JSON object from the file.
    """
    jsonl_lines = []
    with open(file_path, 'r', encoding="utf-8") as file:
        for line in file:
            json_object = json.loads(line)
            jsonl_lines.append(json_object)
            
    return jsonl_lines

def create_query(item):
    """
    Creates the input for the model using the question and the multiple choice options.

    Parameters
    ----------
    item : dict
        A dictionary containing the question and options.
        Expected keys are "question" and "options", where "options" is another
        dictionary with keys "A", "B", "C", and "D".

    Returns
    -------
    str
        A formatted query combining the question and options, ready for use.
    """
    query = f"""## Question {item["question"]}
A. {item["options"]["A"]}             
B. {item["options"]["B"]}
C. {item["options"]["C"]}
D. {item["options"]["D"]}"""
    
    return query

def build_zero_shot_prompt(system_prompt, question):
    """
    Builds the zero-shot prompt.

    Parameters
    ----------
    system_prompt : str
        Task Instruction for the LLM
    question : dict
        The content for which to create a query, formatted as
        required by `create_query`.

    Returns
    -------
    list of dict
        A list of messages, including a system message defining
        the task and a user message with the input question.
    """
    messages = [{"role": "system", "content": system_prompt},
                {"role": "user", "content": create_query(question)}]
    return messages

def format_answer(cot, answer):
    """
    Formats the answer with chain of thought reasoning.

    Parameters
    ----------
    cot : str
        The chain of thought reasoning.
    answer : str
        The final answer choice.

    Returns
    -------
    str
        The formatted answer string.
    """
    return f"""## Answer
{cot}
Therefore, the answer is {answer}"""

def build_few_shot_prompt(system_prompt, question, examples, include_cot=True):
    """
    Builds the few-shot prompt.

    Parameters
    ----------
    system_prompt : str
        Task Instruction for the LLM
    question : dict
        The content for which to create a query, formatted as
        required by `create_query`.
    examples : list of dict
        Examples to include in the prompt.
    include_cot : bool, optional
        Whether to include chain of thought reasoning, by default True.

    Returns
    -------
    list of dict
        A list of messages, including a system message defining
        the task and a user message with the input question.
    """
    messages = [{"role": "system", "content": system_prompt}]
    
    for elem in examples:
        messages.append({"role": "user", "content": create_query(elem)})
        if include_cot:
            messages.append({"role": "assistant", "content": format_answer(elem["cot"], elem["answer_idx"])})        
        else:           
            answer_string = f"""## Answer\nTherefore, the answer is {elem["answer_idx"]}"""
            messages.append({"role": "assistant", "content": answer_string})
            
    messages.append({"role": "user", "content": create_query(question)})
    return messages

def get_response(messages, model_name, temperature=0.0, max_tokens=10):
    """
    Obtains the responses/answers of the model through the chat-completions API.

    Parameters
    ----------
    messages : list of dict
        The built messages provided to the API.
    model_name : str
        Name of the model to access through the API.
    temperature : float, optional
        A value between 0 and 1 that controls the randomness of the output.
        A temperature value of 0 ideally makes the model pick the most likely token, making the outputs deterministic.
    max_tokens : int, optional
        Maximum number of tokens that the model should generate, by default 10.

    Returns
    -------
    str
        The response message content from the model.
    """
    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens
    )
    return response.choices[0].message.content

def matches_ans_option(s):
    """
    Checks if the string starts with the specific pattern 'Therefore, the answer is [A-Z]'.
    
    Parameters
    ----------
    s : str
        The string to be checked.

    Returns
    -------
    bool
        True if the string matches the pattern, False otherwise.
    """
    return bool(re.match(r'^Therefore, the answer is [A-Z]', s))

def extract_ans_option(s):
    """
    Extracts the answer option (a single capital letter) from the start of the string.
    
    Parameters
    ----------
    s : str
        The string containing the answer pattern.

    Returns
    -------
    str or None
        The captured answer option if the pattern is found, otherwise None.
    """
    match = re.search(r'^Therefore, the answer is ([A-Z])', s)
    if match:
        return match.group(1)  # Returns the captured alphabet
    return None 

def matches_answer_start(s):
    """
    Checks if the string starts with the markdown header '## Answer'.
    
    Parameters
    ----------
    s : str
        The string to be checked.

    Returns
    -------
    bool
        True if the string starts with '## Answer', False otherwise.
    """
    return s.startswith("## Answer")

def validate_response(s):
    """
    Validates a multi-line string response that it starts with '## Answer' and ends with the answer pattern.
    
    Parameters
    ----------
    s : str
        The multi-line string response to be validated.

    Returns
    -------
    bool
        True if the response is valid, False otherwise.
    """
    file_content = s.split("\n")
    
    return matches_ans_option(file_content[-1]) and matches_answer_start(s)

def parse_answer(response):
    """
    Parses a response that starts with '## Answer', extracting the reasoning and the answer choice.
    
    Parameters
    ----------
    response : str
        The multi-line string response containing the answer and reasoning.

    Returns
    -------
    tuple
        A tuple containing the extracted CoT reasoning and the answer choice.
    """
    split_response = response.split("\n")
    assert split_response[0] == "## Answer"
    cot_reasoning = "\n".join(split_response[1:-1]).strip()
    ans_choice = extract_ans_option(split_response[-1])
    return cot_reasoning, ans_choice

def get_embedding(text, model="all-minilm:22m"):
    """
    Gets the embedding for the provided text using the specified model.

    Parameters
    ----------
    text : str
        The text for which to get the embedding.
    model : str, optional
        The model to use for generating the embedding, by default "all-minilm:22m".

    Returns
    -------
    list of float
        The embedding vector for the text.
    """
    return client.embeddings.create(input=[text], model=model).data[0].embedding

def find_mode_string_list(string_list):
    """
    Finds the most frequently occurring strings.

    Parameters
    ----------
    string_list : list of str
        A list of strings.

    Returns
    -------
    list of str or None
        A list containing the most frequent string(s) from the input list.
        Returns None if the input list is empty.
    """    
    if not string_list:
        return None  

    string_counts = Counter(string_list)
    max_freq = max(string_counts.values())
    mode_strings = [string for string, count in string_counts.items() if count == max_freq]
    return mode_strings
