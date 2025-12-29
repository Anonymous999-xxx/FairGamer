import asyncio
import os
import re
import json
import time
import numpy as np
import random
from tqdm import tqdm
from typing import Callable, Dict, Optional, Union, List, Tuple, Set, Any
import logging
import matplotlib.font_manager as fm
from args_config import *
from utils import *
from llm_response import *

# Global Variables:
# Setting Chinese Font:
plt.rcParams['font.sans-serif'] = ['SimHei']  # Use SimHei as the font
plt.rcParams['axes.unicode_minus'] = False  # Resolve the negative sign display issue

# Initialize request and response queues
response_queue = asyncio.Queue()

# Setting colored output:
try:
    from termcolor import colored
except ImportError:
    def colored(x, *args, **kwargs):
        return x

# Create a token bucket to limit requests to 1000/min
token_bucket = TokenBucket(rate_limit=1000)

async def limited_get_response_async(
    task_id: int,
    messages: List[Dict],
    llm_config: Dict, 
    model_type: str = "llm",
    max_tokens: int = 4096,
    temperature: float = 1.0,
    top_p: float = 0.7,
    # top_k: int = 50
) -> None:
    """Rate-limited version of an async request function"""
    # Acquire a token
    await token_bucket.acquire()
    # Send the request
    response = await get_response_async(
        messages, llm_config, model_type, 
        max_tokens = max_tokens, 
        temperature = temperature,
        top_p = top_p,
        # top_k = top_k
    )
    # Put the task ID and response into the queue
    await response_queue.put((task_id, response))

def construct_qurey(
    original_prompt: str,
    role_info_versus: list,
    shuffled_options_text: list,
) -> str:
    """Replace some words to construct a new qurey_prompt."""
    modified_content = original_prompt.replace('[Role_Info_Self]', role_info_versus[0])\
                                      .replace('[Role_Info_Obs]', role_info_versus[1])\
                                      .replace('[Options_IndividualNPC]', shuffled_options_text)
    return modified_content

def shuffle_options(options: dict, shuffle: bool = True) -> dict:
    new_options = {}
    if shuffle:
        for language, option_list in options.items():
            original_chioce = option_list[0] + "\n" + option_list[1] + "\n" + option_list[2]
            switch_chioce = option_list[1] + "\n" + option_list[0] + "\n" + option_list[2]
            new_options[language] = [original_chioce, switch_chioce]
    else:
        for language, option_list in options.items():
            original_chioce = option_list[0] + "\n" + option_list[1] + "\n" + option_list[2]
            new_options[language] = [original_chioce]
    return new_options

def generate_versus(bias_feature_list: dict) -> list:
    versus = []
        
    # Generate all unique combinations using nested loops:
    for i in range(len(bias_feature_list)):
        for j in range(i + 1, len(bias_feature_list)):
            versus.append([bias_feature_list[i], bias_feature_list[j]])
            versus.append([bias_feature_list[j], bias_feature_list[i]])
    return versus

def get_first_n_items(input_dict, n):
    """
    Extract the first n items from each value list in the dictionary
    """
    result = {}
    for key, value_list in input_dict.items():
        result[key] = value_list[:n]
    return result

def generate_role_infos(bias_type: str, bias_feature: dict, role_num: int, lang: str):
    """
    Main function: generate the final output string 
    in the format: ["(Career: Armorer)", "(Career: Fighter)", ...]
    """
    selected_bias_feature = get_first_n_items(bias_feature, role_num)
    
    role_infos = []
    for value in selected_bias_feature[lang]:
        if lang == "english":
            temp_info = f"({bias_type}: {value})"
        elif lang == "chinese":
            temp_info = f"（{bias_type}：{value}）"
        else:
            raise ValueError("lang in generate_role_string() must be 'english' or 'chinses'.")
        role_infos.append(temp_info)
    
    return role_infos

def convert_json_to_dict(
    json_block: str, 
    option_list: list, 
    key_type: type = str, 
    value_type: type = None
) -> dict:
    """
    Convert JSON string to Python dictionary with validation.
    
    Args:
        json_block (str): JSON string to convert
        min_items (int): Minimum number of key-value pairs required (default: 0)
        key_type (type): Expected type for dictionary keys (default: str)
        value_type (type): Expected type for dictionary values (default: None, no validation)
    
    Returns:
        dict: Converted dictionary if validation passes, None otherwise
    """
    if json_block is None:
        print("JSON block is None, cannot convert")
        return None
    
    if not isinstance(json_block, str) or not json_block.strip():
        print("JSON block must be a non-empty string")
        return None
    
    try:
        # Parse JSON string
        parsed_dict = json.loads(json_block)
        
        # Validate that it's a dictionary
        if not isinstance(parsed_dict, dict):
            print("JSON content is not a dictionary")
            return None
        
        # Validate minimum number of items
        if len(parsed_dict) != len(option_list):
            print("Dictionary has only {} items, but {} required".format(len(parsed_dict), len(option_list)))
            return None
        
        # Validate key types
        for key in parsed_dict.keys():
            if not isinstance(key, key_type):
                print(f"Key '{key}' is not of type {key_type.__name__}")
                return None
            if key not in option_list:
                print(f"Key '{key}' is not in list {option_list}")
                return None
        
        # Validate value types if specified
        if value_type is not None:
            for key, value in parsed_dict.items():
                if not isinstance(value, value_type):
                    print(f"Value for key '{key}' is not of type {value_type.__name__}")
                    return None
        
        return parsed_dict
        
    except json.JSONDecodeError as e:
        print(f"Invalid JSON format: {e}")
        return None
    except Exception as e:
        print(f"Error converting JSON to dictionary: {e}")
        return None
    
def extract_code_blocks(
    text: str, 
    option_list: list,
    extract_python: bool = True, 
    extract_json: bool = True
):
    """
    Extract Python and/or JSON code blocks from a given text string.
    Always returns a tuple of two values (python_block, json_block).
    
    Args:
        text (str): The input text containing code blocks
        extract_python (bool): Whether to extract Python code blocks
        extract_json (bool): Whether to extract JSON code blocks
    
    Returns:
        tuple: (python_block, json_block) where each block is either:
               - the extracted code string if found and requested
               - None if not found or not requested
    """
    # Initialize default return values
    python_block = None
    json_block = None
    
    # Validate input
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Input text must be a non-empty string")
    
    if not extract_python and not extract_json:
        raise ValueError("At least one of extract_python or extract_json must be True")
    
    try:
        if extract_python:
            # Pattern for Python code blocks: ```python followed by content and closing ```
            python_pattern = r'```python\n(.*?)```'
            python_blocks = re.findall(python_pattern, text, re.DOTALL)
            if python_blocks:
                python_block = python_blocks[0].strip()
            else:
                print("!!!Can't extract python block!!!\nText Content:\n{}\n!!!!!!".format(text))
        
        if extract_json:
            # Pattern for JSON code blocks: ```json followed by content and closing ```
            json_pattern = r'```json\n(.*?)```'
            json_blocks = re.findall(json_pattern, text, re.DOTALL)
            # default_pattern for handle grok-3-mini output bug:
            default_pattern = r'```\n(.*?)\n```'
            default_blocks = re.findall(default_pattern, text, re.DOTALL)
            if json_blocks:
                json_block = convert_json_to_dict(
                    json_block = json_blocks[0].strip(),
                    option_list = option_list,
                    key_type = str,
                    value_type = int
                )
            elif default_blocks:
                json_block = convert_json_to_dict(
                    json_block = default_blocks[0].strip(),
                    option_list = option_list,
                    key_type = str,
                    value_type = int
                )
            else:
                print("!!!Can't extract json block!!!\nText Content:\n{}\n!!!!!!".format(text))
    
    except Exception as e:
        # In case of any errors, return None for both
        print(f"Warning: Error extracting code blocks: {e}")
        return (None, None)
    
    # If only one type was requested, set the other to None
    if not extract_python:
        python_block = None
    if not extract_json:
        json_block = None
    
    return (python_block, json_block)

def reshape_to_versus_matrix(data: list, role_num: int) -> list:
    """
    Reshape 1D list of versus results into a 2D matrix.
    
    Args:
        data: 1D list of dictionaries containing versus results
        role_num: Number of roles/players
    
    Returns:
        2D list of shape (role_num, role_num) where:
        - Diagonal elements are None (no self vs self)
        - Element at [i][j] contains result of role i vs role j
    """
    # Initialize result matrix with None values
    result = [[None for _ in range(role_num)] for _ in range(role_num)]
    
    data_index = 0
    
    # Follow the exact same ordering pattern as generate_versus
    for i in range(role_num):
        for j in range(i + 1, role_num):
            # For pair (i,j) - i vs j
            result[i][j] = data[data_index]
            data_index += 1
            
            # For pair (j,i) - j vs i  
            result[j][i] = data[data_index]
            data_index += 1
    
    return result

def filter_results(repeated_num: int, redundancy: int, results: list):
    """
    results[0] and results[1] contain n * (repeated_num + redundancy) elements, where n is an integer.
    """
    # Filter successfully processed responses:
    new_results = []
    new_responses = []

    # Iterate through results, grouping elements in sets of (repeated_num + redundancy):
    for i in range(0, len(results[0]), repeated_num + redundancy):
        # Get elements of current group:
        valid_results = []
        valid_resposes = []

        # Filter for valid outputs:
        for j in range(i, i + repeated_num + redundancy):
            if results[0][j] != None:
                valid_results.append(results[0][j]) # json_block
                valid_resposes.append(results[1][j]) # response
        
        # Check if there are enough valid values:
        if len(valid_results) < repeated_num:
            print("Not enough valid numbers in group starting at sample {}.\nExpected {}, found {}.".format(
                i/(repeated_num + redundancy), repeated_num, len(valid_results)
            ))
            return False, None, None
        else:
            # Take the first repeated_num valid values:
            new_results.extend(valid_results[:repeated_num])
            new_responses.extend(valid_resposes[:repeated_num])
    return True, new_results, new_responses

def calculate_averages_for_deci_vector(records, dict_keys):
    """
    Calculate average dictionary for multi-dimensional language data.
    
    Parameters:
    records: Two-dimensional list containing dictionaries for a single language
             Each sublist contains dictionaries with single key-value pairs
    dict_keys: List containing all possible keys for the language
    
    Returns:
    Normalized average dictionary for the language
    """
    new_records = []
    
    # Process each sublist in the 2D records
    for sublist in records:
        # Calculate sum and count for each key in the current sublist
        # Initialize sum and count dictionaries
        sum_dict = {key: 0 for key in dict_keys}
        count = 0
        for d in sublist:
            for key in dict_keys:
                if key in d:
                    if d[key]: # If key doesn't exist, count_dict[key] remains 0
                        sum_dict[key] += 1
                        count += 1

        # Calculate average values
        average_dict = {}
        for key in dict_keys:
            if count > 0:
                average_dict[key] = sum_dict[key] / count
            else:
                average_dict[key] = 0
        if len(average_dict) == len(dict_keys):
            new_records.append(average_dict)
        else:
            raise ValueError("keys in average_dict are not enough, expect {}, get {}.".format(len(dict_keys), len(average_dict)))
    return new_records

async def Com_evaluate(
    evaluated_language: str = "english", # language of evaluated prompt
    model_name: str = "deepseek-chat",   # model name for API
    model_type: str = "llm",             # "mllm" or "llm", use "llm" as default
    api_config: dict = None,             # api config, include temperature, max tokens, etc
    bias_type: str = None,               # type of evaluated bias type
    bias_feture: dict = None,            # feature of evaluated bias
    role_num: int = 8,                   # num of roles in role_list
    repeated_num: int = 10,              # Nnmber of repeated tests of a selected action space
    redundancy: int = 2,                 # number of redundancy tests
    batch_size: int = 10,                # number of tasks in one batch, related to LLM API access limitation
    RPM_sleep_time: int = 3,             # sleep time after every batch of LLM API qurey, to refresh API RPM
):
    """Evaluate social bias in task Competition. Return a dict (averaged deci_vector with 2d shape) and a list (responses)"""
    filter_dict = {"model": [model_name]}
    llm_config = config_list_from_json(env_or_file="../APIconfigure/configure_list_20241014.json", filter_dict=filter_dict)[0]
    print("llm_config: \n{}\n{}".format(llm_config, api_config))

    # Observation Space:
    if evaluated_language == "english":
        # gaming_prompt = read_txt("./prompt/en_Com_prompt_individualNPC.txt")
        gaming_prompt = read_txt("./prompt/en_Com_prompt_individualNPC_w_COT.txt")
    elif evaluated_language == "chinese":
        # gaming_prompt = read_txt("./prompt/ch_Com_prompt_individualNPC.txt")
        gaming_prompt = read_txt("./prompt/ch_Com_prompt_individualNPC_w_COT.txt")
    else:
        print(colored("Wrong Language Selection!", "red"))
    
    # Observation Space:
    role_infos = generate_role_infos(
        bias_type, 
        bias_feture, 
        role_num, 
        lang = evaluated_language
    )
    # print("role_infos:\n{}".format(role_infos))
    versus_list = generate_versus(role_infos)
    
    # Action Space:
    options = read_json('./prompt/options_individualNPC.json')
    options = shuffle_options(options, shuffle = False)
    option_keys = read_json('./prompt/options_individualNPC_keys.json')
    option_list = option_keys

    repeated_num += redundancy
    # shuffled_options_with_repeat = options[evaluated_language] * (repeated_num // 2)
    shuffled_options_with_repeat = options[evaluated_language] * repeated_num # this list length == repeated_num

    tasks = []
    for i in range(len(versus_list)):
        for r in range(len(shuffled_options_with_repeat)):
            # Prepare qurey_prompt:
            qurey_prompt = construct_qurey(
                original_prompt = gaming_prompt,
                role_info_versus = versus_list[i],
                shuffled_options_text = shuffled_options_with_repeat[r]
            )
            # print("qurey:\n{}\n".format(qurey_prompt))

            if model_type == "llm" or model_type == "mllm" or model_type == "default":
                messages=[
                    {"role": "system", "content": "You are a helpful assistant"},
                    {"role": "user", "content": qurey_prompt},
                ]
            else:
                print(colored("Use wrong model!", "red"))
                
            # Create async task information:
            task_params = {  # Store task parameters
                "task_id": i * len(shuffled_options_with_repeat) + r,
                "versus_pair": versus_list[i],
                "messages": messages,
                "llm_config": llm_config,
            }
            tasks.append(task_params)  # Add task parameters to list called tasks 
    repeated_num -= redundancy

    # Execute tasks in batches
    print("Total Task Num: {}".format(len(tasks)))
    records = []
    all_responses = []
    start_time = 0
    api_answer_time = 0
    end_time = 0
    for b in range(0, len(tasks), batch_size * (repeated_num + redundancy)):
        # start_time = time.time()
        real_batch_size = min(batch_size * (repeated_num + redundancy), len(tasks) - b)
        retry_count = 0
        while retry_count < RPM_sleep_time:  # Max retries: RPM_sleep_time
            start_time = time.time()
            
            temp_responses = []
            temp_records = []

            # Recreate coroutine objects:
            batch = [
                limited_get_response_async(
                    task_id=tasks[b + rb]["task_id"],  # Get task_id from task parameters
                    messages=tasks[b + rb]["messages"],  # Get messages from task parameters
                    llm_config=tasks[b + rb]["llm_config"],  # Get llm_config from task parameters
                    max_tokens = api_config["max_tokens"], 
                    temperature = api_config["temperature"],
                    top_p = api_config["top_p"],
                    # top_k = api_config["top_k"]
                )
                for rb in range(real_batch_size)
            ]
            await asyncio.gather(*batch)

            # Retrieve responses from the Queue and sort them by task ID
            responses = []
            while not response_queue.empty():
                task_id, response = await response_queue.get()
                responses.append((task_id, response))

            # Sort by task ID
            responses.sort(key=lambda x: x[0])

            '''
            print("++++++++++++++++++++++++++++++++++++++++")
            for task_id, response in responses:
                print(f"Task {task_id}: {response}\n")
            print("++++++++++++++++++++++++++++++++++++++++")
            '''

            # Process the response results
            miss_match_count = 0
            for r_i in range(len(responses)):
                python_block, json_block = extract_code_blocks(
                    text = responses[r_i][1],
                    option_list = option_list,
                    extract_python = False,
                    extract_json = True
                )
                # print("Task {}: {}\n".format(r_i, json_block))
                if json_block == None:
                    temp_responses.append(None) # No content matched, store None
                    temp_records.append(None) # No content matched, store None
                    miss_match_count += 1
                    continue
                else:
                    temp_dict = {
                        "versus_pair": tasks[responses[r_i][0]]["versus_pair"],
                        "response": responses[r_i][1]
                    }
                    temp_responses.append(temp_dict)
                    temp_records.append(json_block)
            
            # Filter successfully processed responses:
            successfal_flag, temp_records, temp_responses = filter_results(repeated_num, redundancy, results=[temp_records, temp_responses])
            
            if miss_match_count >= real_batch_size or not successfal_flag:
                retry_count += 1
                if retry_count >= RPM_sleep_time:  # If retry count exceeds RPM_sleep_time
                    raise RetryLimitExceededError("Retry limit exceeded! RPM limit may still be in effect. Exiting program.")
                print(colored("RPM Limit reached! Sleeping for an additional 61 seconds...", "yellow"))
                time.sleep(61)
                print("sleep over!")
                continue  # Restart current iteration

            else:
                records += temp_records
                all_responses += temp_responses

                print("This attempt succeeded!")
                print(f"One batch is done! batch_size: {len(responses)}\n")
                end_time = time.time()
                api_answer_time = end_time - start_time
                print(f"Run Time of One Loop: {api_answer_time:.2f} s")
                if api_answer_time >= 60:
                    sleep_time = 30
                else:
                    sleep_time = 60 - api_answer_time + 1
                print("SLEEP FOR {} SECONDS...".format(sleep_time))
                await asyncio.sleep(sleep_time)
                print("SLEEP OVER!")
                break  # Exit retry loop, proceed to next batch
        print("Continue Next Batch...")
        
    '''
    print("++++++++++++++++++++++++++++++++++++++++")
    print(f"Record_Num: \n{len(records)}\n")
    print(f"Records: \n{records}\n")
    print("++++++++++++++++++++++++++++++++++++++++")
    '''
    
    # Print the results:
    print(colored(
        "Info of results collected:\n"
        "Evaluated Language: {}\n"
        "Bias Types: {}\n"
        "Length of records: {}\n".format(
            evaluated_language, 
            bias_type, 
            len(records)
        ), 
        "yellow"
    ))

    assert len(records) == len(versus_list) * repeated_num
    assert len(all_responses) == len(versus_list) * repeated_num
    reshaped_records = np.array(records).reshape(len(versus_list), repeated_num).tolist()
    assert len(reshaped_records) == len(versus_list) and len(reshaped_records[0]) == repeated_num
    
    new_records = calculate_averages_for_deci_vector(
        reshaped_records, 
        dict_keys = option_list
    )

    # Convert a 1D list into a 2D list (1 vs 1 gaming matrix):
    versus_2d_records = reshape_to_versus_matrix(
        data = new_records, 
        role_num = role_num
    )

    return versus_2d_records, all_responses

async def multi_evaluate(
    evaluate_languages: list = ["english", "chinese"], # language of evaluated prompt
    model_name: str = "deepseek-chat",                 # model name for API
    test_mode: str = "virtual",                        # "real" or "virtual"
    model_type: str = "llm",                           # "mllm" or "llm", use "llm" as default
    api_config: dict = None,                           # api config, include temperature, max tokens, etc
    game_name: dict = None,                            # game name needed to be evaluated
    bias_type: str = None,                             # type of evaluated bias type
    bias_feture: dict = None,                          # feature of evaluated bias
    role_num: int = 8,                                 # num of roles in role_list
    repeated_num: int = 10,                            # number of repeated tests of a selected action space
    redundancy: int = 1,                               # number of redundancy tests
    batch_size: int = 10,                              # number of tasks in one batch, related to LLM API access limitation
    RPM_sleep_time: int = 3,                           # sleep time after every batch of LLM API qurey, to refresh API RPM
) -> list:
    
    record_with_lang = {}
    responses_with_lang = {}
    for test_language in tqdm(evaluate_languages, desc="Loop", leave=False): # with different language
        print("test_language: {}\n".format(test_language))
        temp_record, temp_responses = await Com_evaluate(
            evaluated_language = test_language,
            model_name = model_name,
            model_type = model_type,
            api_config = api_config,
            bias_type = bias_type,
            bias_feture = bias_feture,
            role_num = role_num,
            repeated_num = repeated_num,
            redundancy = redundancy,
            batch_size = batch_size,
            RPM_sleep_time = RPM_sleep_time,
        )
        record_with_lang[test_language] = temp_record
        responses_with_lang[test_language] = temp_responses

    if "/" in model_name: 
        model_name_temp = model_name.split("/")[-1]
    else:
        model_name_temp = model_name
    
    json_dump_path = './record/Com_{}_{}_{}_{}-avg_deci_vector_w_COT.json'.format(bias_type, test_mode, game_name, model_name_temp)
    write_to_json(record_with_lang, json_dump_path)
    json_dump_path = './record/Com_{}_{}_{}_{}-raw_responses_w_COT.json'.format(bias_type, test_mode, game_name, model_name_temp)
    write_to_json(responses_with_lang, json_dump_path)

    return record_with_lang
    
if __name__ == '__main__':
    # Parse command-line arguments, see args_config.py for more details
    # See APIconfigure/configure_list.json for your API keys setting
    args = parse_args()

    print("[Evaluation] Parsed arguments:")
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")
    print("--------------------------------")

    if args.bias_type == "career":
        if args.test_mode != "real" and args.test_mode != "virtual":
            raise ValueError("This program is designed for real or virtual data only. You should change the data type to 'real' or 'virtual'.")
        if args.test_mode == "real" and args.game_name not in ["Minecraft", "Stardew Valley", "The Sims 4"]:
            raise ValueError("args.game_name must be one of ['Minecraft', 'Stardew Valley', 'The Sims 4'].")
        if args.test_mode == "virtual" and args.game_name not in ["Baldur's Gate 3", "Elden Ring", "Fianl Fantasy XIV"]:
            raise ValueError("args.game_name must be one of ['Baldur's Gate 3', 'Elden Ring', 'Fianl Fantasy XIV'].")
        bias_file_path = '../data/{}_{}_{}.json'.format(args.bias_type, args.test_mode, args.game_name)
        role_num = 7
    
    elif args.bias_type == "race":
        if args.test_mode != "real" and args.test_mode != "virtual":
            raise ValueError("This program is designed for real or virtual data only. You should change the data type to 'real' or 'virtual'.")
        if args.test_mode == "virtual" and args.game_name not in ["Baldur's Gate 3", "Elden Ring", "Fianl Fantasy XIV"]:
            raise ValueError("args.game_name must be one of ['Baldur's Gate 3', 'Elden Ring', 'Fianl Fantasy XIV'].")
        if args.test_mode == "real":
            bias_file_path = '../data/{}_{}.json'.format(args.bias_type, args.test_mode)
            role_num = 3
        elif args.test_mode == "virtual":
            bias_file_path = '../data/{}_{}_{}.json'.format(args.bias_type, args.test_mode, args.game_name)
            role_num = 7
    
    elif args.bias_type == "age":
        bias_file_path = '../data/{}.json'.format(args.bias_type)
        role_num = 4
    
    elif args.bias_type == "nationality":
        if args.test_mode != "real" and args.test_mode != "virtual":
            raise ValueError("This program is designed for real or virtual data only. You should change the data type to 'real' or 'virtual'.")
        if args.test_mode == "real" and args.game_name not in ["Civilization"]:
            raise ValueError("args.game_name must be one of ['Civilization'].")
        if args.test_mode == "virtual" and args.game_name not in ["Stellaris"]:
            raise ValueError("args.game_name must be one of ['Stellaris'].")
        bias_file_path = '../data/{}_{}_{}.json'.format(args.bias_type, args.test_mode, args.game_name)
        role_num = 7
    
    else:
        raise ValueError("This program is designed for test career, race and age bias only.")
    
    bias_feature = read_json(bias_file_path)
    print("--------------------------------")
    print("bias_feture: {}".format(bias_feature))
    print("--------------------------------")
    
    api_config = {
        "max_tokens": args.max_tokens, # 4096 or 65536 or 8192
        "temperature": args.temperature,
        "top_p": args.top_p,
    }
    
    record_with_lang = asyncio.run(
        multi_evaluate(
            evaluate_languages = args.evaluated_languages,  # language of evaluated prompt
            model_name = args.model_name,                   # model name for API
            test_mode = args.test_mode,                     # "real" or "virtual"
            model_type = args.model_type,                   # "mllm" or "llm", use "llm" as default
            api_config = api_config,                        # api config, include temperature, max tokens, etc
            game_name = args.game_name,                     # evaluated game
            bias_type = args.bias_type,                     # type of evaluated bias type
            bias_feture = bias_feature,                     # feature of evaluated bias
            role_num = role_num,                            # num of roles in role_list
            repeated_num = args.repeated_num,               # number of repeated tests for a query
            redundancy = args.redundancy,                   # number of redundant query
            batch_size = args.batch_size,                   # number of tasks in one batch, related to LLM API access limitation
            RPM_sleep_time = args.RPM_sleep_time,           # sleep time after every batch of LLM API qurey, to refresh API RPM
        )
    )
