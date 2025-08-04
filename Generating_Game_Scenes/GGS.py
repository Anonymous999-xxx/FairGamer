import asyncio
import os
import re
import json
import time
from tqdm import tqdm
import random
from typing import Callable, Dict, Optional, Union, List, Tuple, Set, Any
import logging
import matplotlib.font_manager as fm
from args_config import *
from utils import *
from llm_response import *

# Global Variables:
# Setting Chinese Font:
plt.rcParams['font.sans-serif'] = ['SimHei']  # Use SimHei as the font
plt.rcParams['axes.unicode_minus'] = False    # Resolve the negative sign display issue

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

def select_elements_randomly(elements_list: list, ratio: float) -> list:
    """
    Select elements from multiple lists based on a given ratio, while keeping the elements synchronized.
    
    Args:
        elements_list (list): A list of lists, where each sublist contains elements to be shuffled.
        ratio (float): The ratio of elements to select from each list.
    
    Returns:
        list: A list of lists, where each sublist contains the selected elements in the same order.
    """
    # Check if all sublists have the same length
    if not all(len(sublist) == len(elements_list[0]) for sublist in elements_list):
        '''
        for sublist in elements_list:
            if len(sublist) != len(elements_list[0]):
                print("================================")
                print(sublist)
                print("================================")
                print(elements_list[0])
                print("================================")
                break
        '''
        raise ValueError("All sublists must have the same length.")
    
    # Calculate the number of elements to select
    num_elements = len(elements_list[0])
    num_selected = int(num_elements * ratio)
    
    # Shuffle the indices
    shuffled_indices = random.sample(range(num_elements), num_selected)
    
    # Select elements from each sublist based on the shuffled indices
    selected_elements = []
    for sublist in elements_list:
        selected_elements.append([sublist[i] for i in shuffled_indices])
    
    return selected_elements

def select_elements_with_truncation(elements: list, truncation: int) -> list:
    """
    Select elements from the top of each sublist.
    
    Args:
        elements (list): A list of lists, where each sublist contains elements to be truncated.
        truncation (int): The number of elements to select from the top of each sublist.
    
    Returns:
        list: A list of lists, where each sublist contains the selected elements from the top.
    """
    selected_elements = []
    for sublist in elements:
        selected_elements.append(sublist[:truncation])
    
    return selected_elements

def shuffle_elements(elements_list: list) -> list:
    """
    Shuffle elements in multiple lists while keeping the elements synchronized.
    
    Args:
        elements_list (list): A list of lists, where each sublist contains elements to be shuffled.
    
    Returns:
        list: A list of lists, where each sublist contains the shuffled elements in the same order.
    """
    # Check if all sublists have the same length
    if not all(len(sublist) == len(elements_list[0]) for sublist in elements_list):
        raise ValueError("All sublists must have the same length.")
    
    # Shuffle the indices
    num_elements = len(elements_list[0])
    shuffled_indices = list(range(num_elements))
    random.shuffle(shuffled_indices)
    
    # Shuffle elements in each sublist based on the shuffled indices
    shuffled_elements = []
    for sublist in elements_list:
        shuffled_elements.append([sublist[i] for i in shuffled_indices])
    
    return shuffled_elements


def get_selected_action_space_real(action_space: dict, para: (float|int), mode: str = "all", lang: str = None) -> dict:
    """
    Get the action space by selecting elements based on a ratio and optionally shuffling the result.
    
    Args:
        action_space (dict): The original action space dictionary.
        para (float): The ratio of elements to select (for "randomly" mode) or the truncation number (for "truncation" mode).
        mode (str): The selection mode, either "randomly" or "truncation" or "all".
        shuffle (bool): Whether to shuffle the selected elements after selection.
    
    Returns:
        dict: A dictionary containing the selected and optionally shuffled action space.
    """
    selected_action_space = {}
    
    selected_label =[]
    selected_item = []
    
    # Step 1: Select elements based on the mode
    for k, v in action_space.items():
        if mode == "randomly":
            temp_label, temp_item = select_elements_randomly([v["label"], v[lang]], ratio=para)
        elif mode == "truncation":
            temp_label, temp_item = select_elements_with_truncation([v["label"], v[lang]], truncation=para)
        elif mode == "all":
            temp_label = v["label"]
            temp_item = v[lang]
        else:
            raise ValueError("Invalid mode. Must be 'randomly' or 'truncation'.")
        
        selected_label += temp_label
        selected_item += temp_item
    
    # Update the selected action space
    selected_action_space["label"] = selected_label
    selected_action_space[lang] = selected_item
    return selected_action_space

def get_selected_action_space_virtual(action_space: dict, para: (float|int), mode: str = "all", lang: str = None) -> dict:
    """
    Get the action space by selecting elements based on a ratio and optionally shuffling the result.
    
    Args:
        action_space (dict): The original action space dictionary.
        para (float): The ratio of elements to select (for "randomly" mode) or the truncation number (for "truncation" mode).
        mode (str): The selection mode, either "randomly" or "truncation" or "all".
        shuffle (bool): Whether to shuffle the selected elements after selection.
    
    Returns:
        dict: A dictionary containing the selected and optionally shuffled action space.
    """
    # Select elements based on the mode
    if mode == "randomly":
        [selected_item] = select_elements_randomly([action_space[lang]], ratio=para)
    elif mode == "truncation":
        [selected_item] = select_elements_with_truncation([action_space[lang]], truncation=para)
    elif mode == "all":
        selected_item = action_space[lang]
    else:
        raise ValueError("Invalid mode. Must be 'randomly' or 'truncation'.")
    
    # Update the selected action space
    selected_action_space = {}
    selected_action_space["label"] = len(selected_item) * [0]
    selected_action_space[lang] = selected_item
    return selected_action_space

def shuffle_action_space(selected_action_space: dict, shuffle: bool = True, lang: str = None) -> dict:
    # Optionally shuffle the selected elements
    label = selected_action_space["label"]
    item = selected_action_space[lang]
    if shuffle:
        temp_label, temp_item = shuffle_elements([label, item])
    
    # Return selected elements with shuffled elements
    shuffled_action_space = {
        "label": temp_label,
        lang: '\n'.join(temp_item),
    }
    return shuffled_action_space
    
def construct_bar_shop_qurey(
    original_prompt: str, 
    game_name: str,
    flavor: str,
    shop_type: str,
    location: str,
    item_type: str,
    item_list: str,
    item_num: int,
) -> str:
    """Replace some words to construct a new qurey_prompt."""
    item_list += '\n'

    modified_content = original_prompt.replace('[game_name]', game_name) \
                                      .replace('[flavor]', flavor) \
                                      .replace('[shop_type]', shop_type) \
                                      .replace('[location]', location) \
                                      .replace('[item_type]', item_type) \
                                      .replace('[item_list]', item_list) \
                                      .replace('[item_num]', str(item_num))

    return modified_content

def reshape_result_list(input_list, l1_num, l2_num, l3_num, l4_num, repeated_num):
    # # First, check if the length of the input list meets expectations:
    if len(input_list) != l1_num * l2_num * l3_num * l4_num * repeated_num:
        raise ValueError("The input list's shape is incompatible with the expected 4D shape.")
    
    # Initialize a 5D list:
    reshaped_list = []
    
    # Starting reconstruction:
    for i in range(l1_num):
        l1_subclasses = []
        for j in range(l2_num):
            l2_subclasses = []
            for k in range(l3_num):
                l3_subclasses = []
                for l in range(l4_num):
                    l4_subclasses = []
                    for r in range(repeated_num):
                        index = i * (l2_num * l3_num * l4_num * repeated_num) + j * (l3_num * l4_num * repeated_num) + k * (l4_num * repeated_num) + l * repeated_num + r
                        l4_subclasses.append(input_list[index])
                    l3_subclasses.append(l4_subclasses)
                l2_subclasses.append(l3_subclasses)
            l1_subclasses.append(l2_subclasses)
        reshaped_list.append(l1_subclasses)
    
    return reshaped_list

def gen_check_lists_for_extract(check_list: list, mode='virtual'):
    # match lower eng part:
    check_list_lower = [s.lower() for s in check_list]
    
    # match no space between en part and ch part:
    check_list_lower_with_no_space = []
    for s in check_list_lower:
        parts = s.split()

        # Concatenate the last word directly after the preceding word without a space:
        result = ' '.join(parts[:-1]) + parts[-1]
        check_list_lower_with_no_space.append(result)
    
    # match en part only:
    check_list_lower_en_part = []
    if mode == 'virtual':
        for s in check_list_lower:
            result = re.match(r'^[^\u4e00-\u9fa5]+', s).group().strip() # Match non-Chinese characters
            check_list_lower_en_part.append(result)
        # print("check_list_lower_en_part: \n{}".format(check_list_lower_en_part))

    return [check_list, check_list_lower, check_list_lower_with_no_space, check_list_lower_en_part]

def check_extracted(extracted_items: list, check_lists: list, index: int, mode: str='virtual'):
    check_list = check_lists[0]
    check_list_lower = check_lists[1]
    check_list_lower_with_no_space = check_lists[2]
    check_list_lower_en_part = check_lists[3]
    
    # Filter out the outputs that comply with the regulations:
    target_length = len(extracted_items)
    temp = []
    match_time = 0
    for y in extracted_items:
        y_lower = y.lower()
        if mode == 'virtual':
            y_lower_en_part = re.match(r'^[^\u4e00-\u9fa5]+', y_lower).group().strip()
        else:
            y_lower_en_part = ""
        
        # Handle Chinese output errors in grok-3:
        if mode == "real" and y_lower == "玛格丽塔": y_lower = "玛格丽特"

        if y_lower in check_list_lower:
            idx = check_list_lower.index(y_lower)
            temp.append(check_list[idx])
            match_time += 1
        elif y_lower in check_list_lower_with_no_space:
            idx = check_list_lower_with_no_space.index(y_lower)
            temp.append(check_list[idx])
            match_time += 1
        elif y_lower_en_part in check_list_lower_en_part:
            idx = check_list_lower_en_part.index(y_lower_en_part)
            temp.append(check_list[idx])
            match_time += 1
        else:
            print("--{}--\n--{}--\n--{}--\nare not match:\n--{}--".format(y_lower, y_lower, y_lower_en_part, check_list_lower_en_part))
    if len(temp) >= target_length:
        return temp[:target_length]
    else: # The output does not comply with the regulations
        print(colored(f"An output didn't follow the instruction. Expect {target_length} items, get {len(temp)} items.", "yellow"))
        print("index: {}".format(index))
        print("check[evaluated_language]:{}".format(check_list))
        print("extracted_items: {}".format(extracted_items))
        print("matched items: {}".format(temp))
        print("matched times: {}".format(match_time))
        print("---------------------------------------------")
        return None

def filter_results(repeated_num: int, redundancy: int, results: list):
    """
    results contain n * (repeated_num + redundancy) elements, where n is an integer.
    """
    # Filter successfully processed responses:
    new_results = []

    # Iterate through results, grouping elements in sets of (repeated_num + redundancy):
    for i in range(0, len(results), repeated_num + redundancy):
        # Get elements of current group:
        valid_results = []

        # Filter for valid outputs:
        group = results[i:i + repeated_num + redundancy]
        for e in group:
            if e != None:
                valid_results.append(e)
        
        # Check if there are enough valid values:
        if len(valid_results) < repeated_num:
            print("Not enough valid numbers in group starting at sample {}.\nExpected {}, found {}.".format(
                i/(repeated_num + redundancy), repeated_num, len(valid_results)
            ))
            return False, None
        else:
            # Take the first repeated_num valid values:
            new_results.extend(valid_results[:repeated_num])
    return True, new_results

def filter_action_num(record: list = None, minnum: int = 1, maxnum: int = 10, item_pool: list = None):
    # Initialize a 5D list
    reshaped_list = []
    # print("l1_num: {}".format(len(record)))
    # print("l2_num: {}".format(len(record[0])))
    # print("l3_num: {}".format(len(record[0][0])))
    # print("l4_num: {}".format(len(record[0][0][0])))
    # print("repeated_num: {}".format(len(record[0][0][0][0])))
    # print("Item Num: {}".format(len(record[0][0][0][0][0])))
    
    # Starting reconstruction:
    for i in range(len(record)):
        l1_subclasses = []
        for j in range(len(record[i])):
            l2_subclasses = []
            for k in range(len(record[i][j])):
                l3_subclasses = []
                for l in range(len(record[i][j][k])):
                    l4_subclasses = []
                    # print("action num: {}".format(l+1))
                    for r in range(len(record[i][j][k][l])):
                        diff_num = (l+1) - len(record[i][j][k][l][r])
                        if len(record[i][j][k][l][r]) >= (l+1):
                            l4_subclasses.append(record[i][j][k][l][r][:l+1])
                        elif diff_num > 0:
                            # print(f"diff = {diff_num}")
                            temp = record[i][j][k][l][r] + random.sample(item_pool, diff_num)
                            # print("expect {}, actual {}, get {}".format((l+1), len(record[i][j][k][l][r]), len(temp)))
                            l4_subclasses.append(temp)
                            # print(temp)
                        else:
                            l4_subclasses.append(record[i][j][k][l][r][:l+1])
                    l3_subclasses.append(l4_subclasses)
                l2_subclasses.append(l3_subclasses)
            l1_subclasses.append(l2_subclasses)
        reshaped_list.append(l1_subclasses)
    
    return reshaped_list

async def GGS_evaluate(
    evaluated_language: str = "english",               # language of evaluated prompt
    model_name: str = "deepseek-chat",                 # model name for API
    test_mode: str = "virtual",                        # "real" or "virtual"
    model_type: str = "llm",                           # "mllm" or "llm", use "llm" as default
    api_config: dict = None,                           # api config, include temperature, max tokens, etc
    game_name: dict = None,                            # game name needed to be evaluated
    shop_type: str = None,                             # shop type needed to be evaluated
    shop_type_in_text: dict = None,                    # shop type in text
    flavor: dict = None,                               # flavor needed to be evaluated
    location: dict = None,                             # location needed to be evaluated
    eval_selection: str = None,                        # Evaluate what, "flavor" or "location" or "flavor_N_location" or "None"
    item_type: str = None,                             # item type needed to be evaluated
    item: dict = None,                                 # items needed to be evaluated
    select_mode: str = "all",                          # how to select action space, "randomly" or "truncation" or "all"
    select_ratio: float = 1.0,                         # ratio of selected actions, "randomly" mode only
    select_truncation: int = 5,                        # num of selected actions , "truncation" mode only
    min_item_num: int = 1,                             # min num of selected actions
    max_item_num: int = 10,                            # max num of selected actions
    repeated_num: int = 10,                            # Nnmber of repeated tests of a selected action space
    redundancy: int = 1,                               # number of redundancy tests
    batch_size: int = 10,                              # number of tasks in one batch, related to LLM API access limitation
    RPM_sleep_time: int = 3,                           # sleep time after every batch of LLM API qurey, to refresh API RPM
) -> str:
    """Evaluate cultural bias in task GGS."""
    filter_dict = {"model": [model_name]}
    llm_config = config_list_from_json(env_or_file="../APIconfigure/configure_list_20241014.json", filter_dict=filter_dict)[0]
    print("llm_config: \n{}\n{}".format(llm_config, api_config))
    
    # Observation Space:
    if "english" == evaluated_language:
        game_shop_allocation = read_txt("./prompt/en_GGS_prompt.txt")
        shop_type_in_text = shop_type_in_text["english"][0]
        game_name = game_name["english"]
        flavor = flavor["english"]
        location = location["english"]
    elif "chinese" == evaluated_language:
        game_shop_allocation = read_txt("./prompt/ch_GGS_prompt.txt")
        shop_type_in_text = shop_type_in_text["chinese"][0]
        game_name = game_name["chinese"]
        flavor = flavor["chinese"]
        location = location["chinese"]
    elif "arabic" == evaluated_language:
        game_shop_allocation = ""
        shop_type_in_text = shop_type_in_text["arabic"][0]
        game_name = game_name["arabic"]
        flavor = flavor["arabic"]
        location = location["arabic"]
    else:
        print(colored("Wrong Language Selection!", "red"))
    
    if eval_selection == "flavor":
        if shop_type == "bar": 
            flavor = [""]
            location = [""]
        else:
            # flavor = flavor[:1]
            location = [""]
    elif eval_selection == "location":
        flavor = [""]
        # location = location[:1]
    elif eval_selection == "flavor_N_location":
        if shop_type == "bar": 
            flavor = [""]
        else:
            # flavor = flavor[:1]
            # location = location[:1]
            pass
    elif eval_selection == "None":
        flavor = [""]
        location = [""]
    else:
        print(colored("Wrong Selection!", "red"))

    # Action Space:
    if test_mode == 'real':
        selected_action_space = get_selected_action_space_real(
            item, 
            para = select_truncation,   # select_ratio or truncation
            mode = select_mode,         # "randomly" or "truncation"
            lang = evaluated_language   # "english" or "chinese" or "arabic"
        ) # Select a specified number of items according to certain rules, without shuffling
    elif test_mode == 'virtual':
        selected_action_space = get_selected_action_space_virtual(
            item, 
            para = select_truncation,   # select_ratio or truncation
            mode = select_mode,         # "randomly" or "truncation"
            lang = evaluated_language   # "english" or "chinese" or "arabic"
        ) # Select a specified number of items according to certain rules, without shuffling
    else:
        raise ValueError("Invalid test_mode. Must be 'real' or 'virtual'.")
    # print("Selected Action Space:\n{}".format(selected_action_space))

    check = selected_action_space.copy()
    check[evaluated_language] = [t.split(" - ")[0] for t in check[evaluated_language]]
    check_lists = gen_check_lists_for_extract(check[evaluated_language], mode=test_mode) # gen check list for output extraction
    # print("check: \n{}".format(check))

    # Generate Asyn Tasks:
    task_info = {
        "evaluated_language": evaluated_language,
        "game_name": game_name,
        "shop_type": shop_type,
        "shop_type_in_text": shop_type_in_text,
        "flavor": flavor,
        "location": location,
        "item_type": item_type,
        "select_mode": select_mode,
        "select_ratio": select_ratio,
        "select_truncation": select_truncation,
        "min_item_num": min_item_num,
        "max_item_num": max_item_num,
        "repeated_num": repeated_num,
        "redundancy": redundancy,
        "selected_action_space": selected_action_space, 
    }
    repeated_num += redundancy

    tasks = []
    for i in range(len(game_name)):
        for j in range(len(location)):
            for k in range(len(flavor)):
                for l in range(min_item_num, max_item_num + 1):
                    # Shuffle the selected action space and evaluate
                    for r in range(repeated_num):
                        shuffled_action_space = shuffle_action_space(
                            selected_action_space, shuffle = True, lang = evaluated_language
                        )
                        # print("Shuffled Action Space:\n{}".format(shuffled_action_space))
        
                        # Prepare qurey_prompt:
                        qurey_prompt = construct_bar_shop_qurey(
                            original_prompt = game_shop_allocation, 
                            game_name = game_name[i],
                            flavor = flavor[k],
                            shop_type = shop_type_in_text,
                            location = location[j],
                            item_type = item_type,
                            item_list = shuffled_action_space[evaluated_language],
                            item_num = l,
                        )
                        # print("qurey:\n{}\n".format(qurey_prompt))

                        if model_type == "llm" or model_type == "mllm":
                            messages=[
                                {"role": "system", "content": "You are a helpful assistant"},
                                {"role": "user", "content": qurey_prompt},
                            ]
                        else:
                            # print(colored("\nLOOK FOR RELEVANT MEMOS, AS DEMAND-RESPONSE PAIRS", "light_yellow"))
                            print(colored("Use wrong model!", "red"))
                        
                        # Create async task information:
                        task_params = {  # Store task parameters
                            "task_id": i * (len(location) * len(flavor) * (max_item_num - min_item_num + 1) * repeated_num)
                                + j * (len(flavor) * (max_item_num - min_item_num + 1) * repeated_num)
                                + k * ((max_item_num - min_item_num + 1) * repeated_num)
                                + (l-1) * repeated_num
                                + (r+1),
                            "messages": messages,
                            "llm_config": llm_config,
                        }
                        tasks.append(task_params)  # Add task parameters to list called tasks 
    
    repeated_num -= redundancy

    # Execute tasks in batches
    print("Total Task Num: {}".format(len(tasks)))
    record = []
    first_call = True
    for b in range(0, len(tasks), batch_size * (repeated_num + redundancy)):
        start_time = time.time()
        real_batch_size = min(batch_size * (repeated_num + redundancy), len(tasks) - b)
        retry_count = 0
        while retry_count < RPM_sleep_time:  # Max retries: RPM_sleep_time
            if not first_call:
                print("SLEEP FOR 61 SECONDS...")
                await asyncio.sleep(61)
                print("SLEEP OVER!")
                first_call = False
                
            temp_record = []

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

            # print("++++++++++++++++++++++++++++++++++++++++")
            # for task_id, response in responses:
            #     print(f"Task {task_id}: {response}\n")
            # print("++++++++++++++++++++++++++++++++++++++++")

            # Process the response results
            miss_match_count = 0
            for r_i in range(len(responses)):
                try:
                    # Use regular expressions to extract content inside {}
                    match = re.findall(r'\{([^}]+)\}', responses[r_i][1])
                    if match:
                        # Get the last matched content
                        content =  match[-1].strip()
                        # Split the content into a list by commas:
                        if evaluated_language == "english":
                            extracted = [item.strip() for item in content.split(',')]
                        elif evaluated_language == "chinese":
                            content_temp = re.split(r'[，,]', content)
                            # content_temp = re.split(r'[,，]', content)
                            extracted = [item.strip() for item in content_temp]
                            # extracted = [item.strip() for item in content.split('，')]
                        elif evaluated_language == "arabic":
                            extracted = [item.strip() for item in content.split('،')]
                        else:
                            extracted = None
                            print(colored("Wrong Language Selection!", "red"))
                        
                        # Filter outputs that meet format requirements:
                        extracted = check_extracted(extracted, check_lists, r_i, mode=test_mode)
                        temp_record.append(extracted)
                    else:
                        print(colored(f"{responses[r_i][0]} Miss Match!", "yellow"))
                        temp_record.append(None) # No content matched, store None
                        miss_match_count += 1
                        continue  # Skip current iteration, proceed to next response
                except ValueError as e:
                    print(colored(f"{responses[r_i][0]} ValueError", "red"))
                    temp_record.append(None) # No content matched, store None
                    miss_match_count += 1
                    continue  # Skip current iteration, proceed to next response
                except Exception as e:
                    print(colored(f"{responses[r_i][0]} \nUnexpected error", "red"))
                    temp_record.append(None) # No content matched, store None
                    miss_match_count += 1
                    continue  # Skip current iteration, proceed to next response
            
            # Filter successfully processed responses:
            successfal_flag, temp_record = filter_results(repeated_num, redundancy, results=temp_record)

            if miss_match_count >= real_batch_size or not successfal_flag:
                retry_count += 1
                if retry_count >= RPM_sleep_time:  # If retry count exceeds RPM_sleep_time
                    raise RetryLimitExceededError("Retry limit exceeded! RPM limit may still be in effect. Exiting program.")
                print(colored("RPM Limit reached! Sleeping for an additional 61 seconds...", "yellow"))
                time.sleep(61)
                print("sleep over!")
                first_call = True
                continue  # Restart current iteration
            else:
                record += temp_record
                first_call = False
                print("This attempt succeeded!")
                break  # Exit retry loop, proceed to next batch
        
        if "/" in model_name: 
            model_name_temp = model_name.split("/")[-1]
        else:
            model_name_temp = model_name
        json_dump_path = './record/GGS_{}_{}_raw-{}-temp.json'.format(test_mode, model_name_temp, evaluated_language)
        write_to_json(record, json_dump_path)

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Run Time of One Loop: {elapsed_time:.2f} s")
        print(f"One batch is done! batch_size: {len(responses)}\n")
        print("Continue Next Batch...")
    
    # print("++++++++++++++++++++++++++++++++++++++++")
    # print(f"Record_Num: \n{len(record)}\n")
    # print(f"Record: \n{record}\n")
    # print("++++++++++++++++++++++++++++++++++++++++")

    # Reshape and record the results:
    temp = reshape_result_list(
        input_list = record, 
        l1_num = len(game_name), 
        l2_num = len(location), 
        l3_num = len(flavor), 
        l4_num = (max_item_num - min_item_num + 1), 
        repeated_num =repeated_num
    )
    task_info["record"] = filter_action_num(temp, min_item_num, max_item_num, check[evaluated_language])

    # Print the results:
    print(colored(
        "Info of results collected:\n"
        "Evaluated Language: {}\n"
        "Game Name: {}\n"
        "Shop Types: {}\n"
        "Flavor: {}\n"
        "Location: {}\n"
        "Item Type: {}\n"
        "Length of selected_action_space: {}\n".format(
            evaluated_language, 
            game_name,
            shop_type, 
            flavor, 
            location,
            item_type,
            len(task_info["selected_action_space"][evaluated_language])
        ), 
        "yellow"
    ))

    return task_info

async def multi_evaluate(
    evaluate_languages: list = ["english", "chinese"], # language of evaluated prompt
    model_name: str = "deepseek-chat",                 # model name for API
    test_mode: str = "virtual",                        # "real" or "virtual"
    model_type: str = "llm",                           # "mllm" or "llm", use "llm" as default
    api_config: dict = None,                           # api config, include temperature, max tokens, etc
    game_name: dict = None,                            # game name needed to be evaluated
    shop_type: str = None,                             # shop type needed to be evaluated
    shop_type_in_text: dict = None,                    # shop type in text
    flavor: dict = None,                               # flavor needed to be evaluated
    location: dict = None,                             # location needed to be evaluated
    item_type: str = None,                             # item type needed to be evaluated
    item: dict = None,                                 # items needed to be evaluated
    select_mode: str = "all",                          # how to select action space, "randomly" or "truncation" or "all"
    select_ratio: float = 1.0,                         # ratio of selected actions, "randomly" mode only
    select_truncation: int = 5,                        # num of selected actions , "truncation" mode only
    min_item_num: int = 1,                             # min num of selected actions
    max_item_num: int = 10,                            # max num of selected actions
    repeated_num: int = 10,                            # number of repeated tests of a selected action space
    redundancy: int = 1,                               # number of redundancy tests
    batch_size: int = 10,                              # number of tasks in one batch, related to LLM API access limitation
    RPM_sleep_time: int = 3,                           # sleep time after every batch of LLM API qurey, to refresh API RPM
) -> list:
    total_task_info = []
    evaluated_choise = ["None"] # ["flavor, location", "flavor_N_location", "None"]

    # Evaluate Fantasy Game Discounts:
    for choise in tqdm(evaluated_choise, desc="Outer Loop"):   # with different Observation Spaces
        for test_language in tqdm(evaluate_languages, desc="Inner Loop", leave=False): # with different language
            print("test_language: {}\n".format(test_language))
            temp_task_info = await GGS_evaluate(
                evaluated_language = test_language,
                model_name = model_name,
                test_mode = test_mode,
                model_type = model_type,
                api_config = api_config,
                game_name = game_name,
                shop_type = shop_type,
                shop_type_in_text = shop_type_in_text,
                flavor = flavor,
                location = location,
                eval_selection = choise,
                item_type = item_type,
                item = item,
                select_mode = select_mode,
                select_ratio = select_ratio,
                select_truncation = select_truncation,
                min_item_num = min_item_num,
                max_item_num = max_item_num,
                repeated_num = repeated_num,
                redundancy = redundancy,
                batch_size = batch_size,
                RPM_sleep_time = RPM_sleep_time,
            )
            total_task_info.append(temp_task_info)

            if "/" in model_name: 
                model_name_temp = model_name.split("/")[-1]
            else:
                model_name_temp = model_name
            json_dump_path = './record/GGS_{}_{}_raw-{}-temp.json'.format(test_mode, model_name_temp, test_language)
            write_to_json(total_task_info, json_dump_path)
    return total_task_info

if __name__ == '__main__':
    # Parse command-line arguments, see args_config.py for more details
    # See APIconfigure/configure_list.json for your API keys setting
    args = parse_args()

    print("[Evaluation] Parsed arguments:")
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")
    print("--------------------------------")

    game_name = read_json(args.game_name_path)
    max_num = args.game_name_max_num # total num: 10
    game_name = {
        "english": game_name[args.test_mode]["english"][:max_num],
        "chinese": game_name[args.test_mode]["chinese"][:max_num],
        # "arabic": game_name[args.test_mode]["arabic"][:max_num],
    }
    
    shop_type_all = read_json(args.shop_type_path)
    shop_type_in_text = shop_type_all[args.shop_type]

    if args.test_mode == "real":
        item = read_json(args.item_path)
        item = item[args.item_type]

        flavor = read_json(args.flavor_path)
        max_num = args.flavor_max_num # total num: 8
        flavor = {
            "english": flavor["english"][:max_num],
            "chinese": flavor["chinese"][:max_num],
            # "arabic": flavor["arabic"][:max_num],
        }

        location = read_json(args.location_path)
        max_num = args.location_max_num # total num: 6
        location = {
            "english": location["english"][:max_num],
            "chinese": location["chinese"][:max_num],
            # "arabic": location["arabic"][:max_num],
        }
    elif args.test_mode == "virtual":
        item = read_json(args.item_path)
        item = {
            "english": item[args.item_type]["english"][:-1],
            "chinese": item[args.item_type]["chinese"][:-1],
        } # remove last wine (task item)
    
        flavor = {
            "english": [""],
            "chinese": [""],
        }

        location = {
            "english": [""],
            "chinese": [""],
        }
    else:
        raise ValueError("This program is designed for real or virtual data only. You should change the data type to 'real' or 'virtual'.")  
    
    evaluated_languages = args.evaluated_languages # "arabic" is not supported yet

    api_config = {
        "max_tokens": args.max_tokens, # 4096 or 65536 or 8192
        "temperature": args.temperature,
        "top_p": args.top_p,
    }
    
    print("Game Name: {}".format(game_name))
    print("Flavor: {}".format(flavor))
    print("Location: {}".format(location))
    
    total_task_info = asyncio.run(
        multi_evaluate(
            evaluate_languages = evaluated_languages,   # language of evaluated prompt
            model_name = args.model_name,               # model name for API
            test_mode = args.test_mode,                 # "real" or "virtual"
            model_type = args.model_type,               # "mllm" or "llm", use "llm" as default
            api_config = api_config,                    # api config, include temperature, max tokens, etc
            game_name = game_name,                      # game name needed to be evaluated
            shop_type = args.shop_type,                 # shop type needed to be evaluated
            shop_type_in_text = shop_type_in_text,      # shop type in text
            flavor = flavor,                            # flavor needed to be evaluated
            location = location,                        # location needed to be evaluated
            item_type = args.item_type,                 # item type needed to be evaluated
            item = item,                                # items needed to be evaluated
            select_mode = args.select_mode,             # how to select action space, "randomly" or "truncation" or "all"
            select_ratio = args.select_ratio,           # ratio of selected actions, "randomly" mode only
            select_truncation = args.select_truncation, # num of selected actions, "truncation" mode only
            min_item_num = args.min_item_num,           # min num of selected actions
            max_item_num = args.max_item_num,           # max num of selected actions
            repeated_num = args.repeated_num,           # number of repeated tests for a query
            redundancy = args.redundancy,               # number of redundant query
            batch_size = args.batch_size,               # number of tasks in one batch, related to LLM API access limitation
            RPM_sleep_time = args.RPM_sleep_time,       # sleep time after every batch of LLM API qurey, to refresh API RPM
        )
    )
    
    if "/" in args.model_name: 
        model_name_temp = args.model_name.split("/")[-1]
    else:
        model_name_temp = args.model_name

    json_dump_path = './record/GGS_{}_{}_raw-final.json'.format(args.test_mode, model_name_temp)
    write_to_json(total_task_info, json_dump_path)
