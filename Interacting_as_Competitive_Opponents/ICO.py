import asyncio
import os
import re
import json
import time
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

def shuffle_options(options: dict) -> dict:
    new_options = {}
    for language, option_list in options.items():
        original_chioce = option_list[0] + "\n" + option_list[1] + "\n" + option_list[2] + "\n"
        switch_chioce = option_list[0] + "\n" + option_list[2] + "\n" + option_list[1] + "\n"
        new_options[language] = [original_chioce, switch_chioce]
    return new_options

def generate_versus(country: dict) -> dict:
    versus = {}
    for language, option_list in country.items():
        versus[language] = []
        
        # Generate all unique combinations using nested loops:
        for i in range(len(option_list)):
            for j in range(i + 1, len(option_list)):
                versus[language].append([option_list[i], option_list[j]])
                versus[language].append([option_list[j], option_list[i]])
    return versus

def construct_diplomatic_game_qurey(
    original_prompt: str, 
    game_name: str,
    versus_pair: list,
    diplomatic_options: str,
) -> str:
    """Replace some words to construct a new qurey_prompt."""
    modified_content = original_prompt.replace('[game_name]', game_name) \
                                      .replace('[country_name_1]', versus_pair[0]) \
                                      .replace('[country_name_2]', versus_pair[1]) \
                                      .replace('[diplomatic_options]', diplomatic_options)
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

def filter_and_trans_results(repeated_num: int, redundancy: int, results: list, trans_list: list):
    """
    results contain n * (repeated_num + redundancy) elements, where n is an integer.
    """
    # Filter successfully processed responses:
    new_results = []

    # Iterate through results, grouping elements in sets of (repeated_num + redundancy):
    for i in range(0, len(results), repeated_num + redundancy):
        # Get elements of current group:
        group = results[i:i + repeated_num + redundancy]
    
        # Filter for valid outputs:
        valid_numbers = []
        for x in group:
            if x == None:
                pass # Output does not comply with regulations
            elif x.lower() == trans_list[0].lower(): # Neutral
                valid_numbers.append(0)
            elif x.lower() == trans_list[1].lower(): # Oppose
                valid_numbers.append(1)
            elif x.lower() == trans_list[2].lower(): # Cooperate
                valid_numbers.append(2)
            else:
                pass # Output does not comply with regulations

        # Check if there are enough valid values:
        if len(valid_numbers) < repeated_num:
            print("Not enough valid numbers in group starting at index {}. \nExpected {}, found {}.".format(
                i/(repeated_num + redundancy), repeated_num, len(valid_numbers)
            ))
            return False, None
    
        # Take the first repeated_num valid values:
        new_results.extend(valid_numbers[:repeated_num])
    return True, new_results
        

async def ICO_evaluate(
    evaluated_language: str = "english", # language of evaluated prompt
    model_name: str = "deepseek-chat",   # model name for API
    test_mode: str = "virtual",          # "real" or "virtual"
    model_type: str = "llm",             # "mllm" or "llm", use "llm" as default
    api_config: dict = None,             # api config, include temperature, max tokens, etc
    game_name: dict = None,              # game name needed to be evaluated
    country: dict = None,                # countries needed to be evaluated
    repeated_num: int = 10,              # Nnmber of repeated tests of a selected action space
    redundancy: int = 1,                 # number of redundancy tests
    batch_size: int = 10,                # number of tasks in one batch, related to LLM API access limitation
    RPM_sleep_time: int = 3,             # sleep time after every batch of LLM API qurey, to refresh API RPM
) -> dict:
    """Evaluate cultural bias in task ICO, especially in diplomatic games."""
    filter_dict = {"model": [model_name]}
    llm_config = config_list_from_json(env_or_file="../APIconfigure/configure_list_20241014.json", filter_dict=filter_dict)[0]
    print("llm_config: \n{}\n{}".format(llm_config, api_config))

    # Observation Space:
    game_name = game_name[evaluated_language]
    versus = generate_versus(country)
    versus_list = versus[evaluated_language] # country versus list

    if "english" == evaluated_language:
        diplomatic_advantage = read_txt("./prompt/en_diplomatic_advantage.txt")
        diplomatic_disadvantage = read_txt("./prompt/en_diplomatic_disadvantage.txt")
        diplomatic_prompts = [diplomatic_advantage, diplomatic_disadvantage]
    elif "chinese" == evaluated_language:
        diplomatic_advantage = read_txt("./prompt/ch_diplomatic_advantage.txt")
        diplomatic_disadvantage = read_txt("./prompt/ch_diplomatic_disadvantage.txt")
        diplomatic_prompts = [diplomatic_advantage, diplomatic_disadvantage]
    else:
        print(colored("Wrong Language Selection!", "red"))
    
    # Action Space:
    diplomatic_options = shuffle_options(
        read_json('./data/diplomatic_options.json')
    )
    diplomatic_options = diplomatic_options[evaluated_language]
    choice_to_number = read_json('./data/choice_to_number.json')
    
    # Generate Asyn Tasks:
    record = []
    task_info = {
        "evaluated_language": evaluated_language,
        "game_name": game_name,
        "country_num": len(country[evaluated_language]),
        "country": country[evaluated_language],
        "versus_num": len(versus_list),
        "versus": versus_list,
        "choices": choice_to_number[evaluated_language],
        "prompt_num": len(diplomatic_prompts),
        "diplomatic_option_num": len(diplomatic_options),
        "repeated_num": repeated_num,
        "redundancy": redundancy,
    }

    repeated_num += redundancy
    tasks = []
    for gn in range(len(game_name)):
        for i in range(len(versus_list)):
            for j in range(len(diplomatic_prompts)):
                for k in range(len(diplomatic_options)):
                    # Prepare qurey_prompt:
                    qurey_prompt = construct_diplomatic_game_qurey(
                        original_prompt = diplomatic_prompts[j], 
                        game_name = game_name[gn],
                        versus_pair = versus_list[i],
                        diplomatic_options = diplomatic_options[k]
                    )
                    # print("qurey:\n{}\n".format(qurey_prompt))

                    for r in range(repeated_num):
                        if model_type == "llm" or model_type == "mllm" or model_type == "default":
                            messages=[
                                {"role": "system", "content": "You are a helpful assistant"},
                                {"role": "user", "content": qurey_prompt},
                            ]
                        else:
                            print(colored("Use wrong model!", "red"))
                
                        # Create async task information:
                        task_params = {  # Store task parameters
                            "task_id": gn * (len(game_name) * len(diplomatic_prompts) * len(diplomatic_options) * repeated_num) 
                                + i * (len(diplomatic_prompts) * len(diplomatic_options) * repeated_num) 
                                + j * (len(diplomatic_options) * repeated_num) 
                                + k * repeated_num 
                                + r + 1,
                            "messages": messages,
                            "llm_config": llm_config,
                        }
                        tasks.append(task_params)  # Add task parameters to list called tasks 
    
    repeated_num -= redundancy

    # Execute tasks in batches
    print("Total Task Num: {}".format(len(tasks)))
    record = []
    first_call = True
    start_time = 0
    api_answer_time = 0
    end_time = 0
    for b in range(0, len(tasks), batch_size * (repeated_num + redundancy)):
        # start_time = time.time()
        real_batch_size = min(batch_size * (repeated_num + redundancy), len(tasks) - b)
        retry_count = 0
        while retry_count < RPM_sleep_time:  # Max retries: RPM_sleep_time
            if api_answer_time - start_time >= 60:
                sleep_time = 30
            else:
                sleep_time = 60-(api_answer_time - start_time) + 1
            if not first_call:
                print("SLEEP FOR {} SECONDS...".format(sleep_time))
                await asyncio.sleep(sleep_time)
                print("SLEEP OVER!")
                first_call = False
            
            start_time = time.time()
            
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
                        content = match[-1].strip()
                        if ":" in content: content_type2 = content.split(":")[1].strip().lower()
                        if "：" in content: content_type3 = content.split("：")[1].strip()

                        # 过滤出符合格式要求的输出:
                        if content.lower() in [item.lower() for item in choice_to_number[evaluated_language]]:
                            temp_record.append(content)  # successfully processed and stored the content
                            # print("---------extracted selection:---------")
                            # print(content)
                            # print("--------------------------------------")
                        elif ":" in content and content_type2 in [item.lower() for item in choice_to_number[evaluated_language]]:
                            temp_record.append(content_type2)  # successfully processed and stored the content
                            # print("---------extracted selection:---------")
                            # print("--" + content_type2+"--")
                            # print("--------------------------------------")
                        elif "：" in content and content_type3 in [item for item in choice_to_number[evaluated_language]]:
                            temp_record.append(content_type3)  # successfully processed and stored the content
                            # print("---------extracted selection:---------")
                            # print("--" + content_type3 + "--")
                            # print("--------------------------------------")
                        else:
                            print(colored(f"{responses[r_i][0]} Output Format Error!", "yellow"))
                            temp_record.append(None) # No content matched, store None
                            miss_match_count += 1
                            print("content: {}\n".format(content))
                            print("choice_to_number{}: {}\n".format(evaluated_language, choice_to_number[evaluated_language]))
                            print("----------")
                            print("response[{}]:\n{}".format(responses[r_i][0], responses[r_i][1]))
                            print("----------")
                    else:
                        print(colored(f"{responses[r_i][0]} Miss Match!", "yellow"))
                        temp_record.append(None) # No content matched, store Nonee
                        miss_match_count += 1
                        continue  # Skip current iteration, proceed to next response
                except ValueError as e:
                    print(colored(f"{responses[r_i][0]} ValueError", "red"))
                    temp_record.append(None) # No content matched, store None
                    miss_match_count += 1
                    continue  # Skip current iteration, proceed to next response
                except Exception as e:
                    print(colored(f"{responses[r_i][0]} Unexpected error", "red"))
                    temp_record.append(None) # No content matched, store None
                    miss_match_count += 1
                    continue  # Skip current iteration, proceed to next response
            
            # Filter successfully processed responses:
            successfal_flag, temp_record = filter_and_trans_results(
                repeated_num, 
                redundancy, 
                results=temp_record, 
                trans_list=choice_to_number[evaluated_language]
            )
            
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
                api_answer_time = time.time()
                record += temp_record
                first_call = False
                print("This attempt succeeded! Time cost: {} s".format(api_answer_time - start_time))
                break  # Exit retry loop, proceed to next batch
        
        if "/" in model_name: 
            model_name_temp = model_name.split("/")[-1]
        else:
            model_name_temp = model_name
        json_dump_path = './record/ICO_{}_{}_raw-{}-temp.json'.format(test_mode, model_name_temp, evaluated_language)
        write_to_json(record, json_dump_path)

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Run Time of One Loop: {elapsed_time:.2f} s")
        print(f"One batch is done! batch_size: {len(responses)}\n")
        print("Continue Next Batch...")

        #-----------------------------------------------------
        
    # print("++++++++++++++++++++++++++++++++++++++++")
    # print(f"Record_Num: \n{len(record)}\n")
    # print(f"Record: \n{record}\n")
    # print("++++++++++++++++++++++++++++++++++++++++")

    # Reshape and record the results:
    task_info["record"] = reshape_result_list(
        input_list = record, 
        l1_num = len(game_name),
        l2_num = len(versus_list), 
        l3_num = len(diplomatic_prompts), 
        l4_num = len(diplomatic_options), 
        repeated_num =repeated_num
    )

    mes = colored(
        "Info of results collected:\n" +
        "Evaluated Language: {}\n".format(evaluated_language) +
        "Number of Versus Combinations: {}\n".format(len(versus_list)) +
        "Number of Diplomatic Situations: {}\n".format(len(diplomatic_prompts)) +
        "Number of Combinations of Diplomatic Options: {}".format(len(diplomatic_options)),
        "yellow"
    )
    print(mes)

    return task_info

async def multi_evaluate(
    evaluate_languages: list = ["english", "chinese"], # language of evaluated prompt
    model_name: str = "deepseek-chat",                 # model name for API
    test_mode: str = "virtual",                        # "real" or "virtual"
    model_type: str = "llm",                           # "mllm" or "llm", use "llm" as default
    api_config: dict = None,                           # api config, include temperature, max tokens, etc
    game_name: dict = None,                            # game name needed to be evaluated
    country: dict = None,                              # countries needed to be evaluated
    repeated_num: int = 10,                            # number of repeated tests of a selected action space
    redundancy: int = 1,                               # number of redundancy tests
    batch_size: int = 10,                              # number of tasks in one batch, related to LLM API access limitation
    RPM_sleep_time: int = 3,                           # sleep time after every batch of LLM API qurey, to refresh API RPM
) -> list:
    total_task_info = []

    # Evaluate Fantasy Game Discounts:
    for test_language in tqdm(evaluate_languages, desc="Outer Loop"): # with different language
        print("test_language: {}".format(test_language))
        temp_task_info = await ICO_evaluate(
            evaluated_language = test_language,
            model_name = model_name,
            test_mode = test_mode,
            model_type = model_type,
            api_config = api_config,
            game_name = game_name,
            country = country,
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
        json_dump_path = './record/ICO_{}_{}_raw-{}-temp.json'.format(test_mode, model_name_temp, test_language)
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
        "arabic": game_name[args.test_mode]["arabic"][:max_num],
    }
    
    country = read_json(args.country_path)
    country_num = args.country_max_num
    country = {
        "english": country[args.test_mode]["english"][:country_num],
        "chinese": country[args.test_mode]["chinese"][:country_num],
        "arabic": country[args.test_mode]["arabic"][:country_num],
    }
    # country = country_selection(country, target_num = 10)

    evaluated_languages = args.evaluated_languages # "arabic" is not supported yet
    
    api_config = {
        "max_tokens": args.max_tokens, # 4096 or 65536 or 8192
        "temperature": args.temperature,
        "top_p": args.top_p,
    }

    print("Game Name: {}".format(game_name))
    print("Country: {}".format(country))

    total_task_info = asyncio.run(
        multi_evaluate(
            evaluate_languages = evaluated_languages,   # language of evaluated prompt
            model_name = args.model_name,               # model name for API
            test_mode = args.test_mode,                 # "real" or "virtual"
            model_type = args.model_type,               # "mllm" or "llm", use "llm" as default
            api_config = api_config,                    # api config, include temperature, max tokens, etc
            game_name = game_name,                      # game name needed to be evaluated
            country = country,                          # countries needed to be evaluated
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

    json_dump_path = './record/ICO_{}_{}_raw-final.json'.format(args.test_mode, model_name_temp)
    write_to_json(total_task_info, json_dump_path)
