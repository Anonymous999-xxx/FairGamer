import asyncio
import os
import re
import json
from tqdm import tqdm
from args_config import *
from utils import *
from llm_response import *
from Cooperation import *

import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
from mcv import *

if __name__ == '__main__':
    # Parse command-line arguments, see args_config.py for more details
    # See APIconfigure/configure_list.json for your API keys setting
    game_name_list_1 = ["Minecraft", "Stardew Valley", "The Sims 4"]
    game_name_list_2 = ["Baldur's Gate 3", "Elden Ring", "Fianl Fantasy XIV"]
    game_name_list_3 = ["Civilization"]
    game_name_list_4 = ["Stellaris"]
    bias_type_list = ["nationality", "career", "race", "age"]
    # bias_type_list = ["career", "race", "age"]
    test_mode_list = ["real", "virtual"]
    AP_num = 100
    args = parse_args()
    

    args_list = []
    for bias_type in bias_type_list:
        if bias_type == "age":
            new_args = argparse.Namespace(**vars(args))
            new_args.bias_type = bias_type
            new_args.game_name = None
            new_args.test_mode = None
            new_args.bias_file_path = '../data/{}.json'.format(bias_type)
            new_args.role_num = 4
            args_list.append(new_args)
        elif bias_type == "nationality":
            for test_mode in test_mode_list:
                if test_mode == "real":
                    for game_name in game_name_list_3:
                        new_args = argparse.Namespace(**vars(args))
                        new_args.bias_type = bias_type
                        new_args.game_name = game_name
                        new_args.test_mode = test_mode
                        new_args.bias_file_path = '../data/{}_{}_{}.json'.format(new_args.bias_type, new_args.test_mode, new_args.game_name)
                        new_args.role_num = 7
                        args_list.append(new_args)
                elif test_mode == "virtual":
                    for game_name in game_name_list_4:
                        new_args = argparse.Namespace(**vars(args))
                        new_args.bias_type = bias_type
                        new_args.game_name = game_name
                        new_args.test_mode = test_mode
                        new_args.bias_file_path = '../data/{}_{}_{}.json'.format(new_args.bias_type, new_args.test_mode, new_args.game_name)
                        new_args.role_num = 7
                        args_list.append(new_args)
                else:
                    raise ValueError('Wrong test_mode when bias_type == "nationality".')
        elif bias_type == "career":
            for test_mode in test_mode_list:
                if test_mode == "real":
                    for game_name in game_name_list_1:
                        new_args = argparse.Namespace(**vars(args))
                        new_args.bias_type = bias_type
                        new_args.test_mode = test_mode
                        new_args.game_name = game_name
                        new_args.bias_file_path = '../data/{}_{}_{}.json'.format(new_args.bias_type, new_args.test_mode, new_args.game_name)
                        new_args.role_num = 7
                        args_list.append(new_args)
                elif test_mode == "virtual":
                    for game_name in game_name_list_2:
                        new_args = argparse.Namespace(**vars(args))
                        new_args.bias_type = bias_type
                        new_args.test_mode = test_mode
                        new_args.game_name = game_name
                        new_args.bias_file_path = '../data/{}_{}_{}.json'.format(new_args.bias_type, new_args.test_mode, new_args.game_name)
                        new_args.role_num = 7
                        args_list.append(new_args)
                else:
                    raise ValueError('Wrong test_mode when bias_type == "career".')
        elif bias_type == "race":
            for test_mode in test_mode_list:
                if test_mode == "real":
                    new_args = argparse.Namespace(**vars(args))
                    new_args.bias_type = bias_type
                    new_args.test_mode = test_mode
                    new_args.game_name = None
                    new_args.bias_file_path = '../data/{}_{}.json'.format(new_args.bias_type, new_args.test_mode)
                    new_args.role_num = 3
                    args_list.append(new_args)
                elif test_mode == "virtual":
                    for game_name in game_name_list_2:
                        new_args = argparse.Namespace(**vars(args))
                        new_args.bias_type = bias_type
                        new_args.test_mode = test_mode
                        new_args.game_name = game_name
                        new_args.bias_file_path = '../data/{}_{}_{}.json'.format(new_args.bias_type, new_args.test_mode, new_args.game_name)
                        new_args.role_num = 7 # 8
                        args_list.append(new_args)
                else:
                    raise ValueError('Wrong test_mode when bias_type == "race".')
        else:
            raise ValueError('Wrong bias_type .')
    
    analysis_list = []
    for i in range(len(args_list)):
        print("[Evaluation] Parsed {}-th arguments:".format(i+1))
        for arg in vars(args_list[i]):
            print(f"{arg}: {getattr(args_list[i], arg)}")
        print("--------------------------------")
    
        api_config = {
            "max_tokens": args_list[i].max_tokens, # 4096 or 65536 or 8192
            "temperature": args_list[i].temperature,
            "top_p": args_list[i].top_p,
        }
        bias_feature = read_json(args_list[i].bias_file_path)

        record_with_lang = asyncio.run(
            multi_evaluate(
                evaluate_languages = args_list[i].evaluated_languages,   # language of evaluated prompt
                model_name = args_list[i].model_name,               # model name for API
                test_mode = args_list[i].test_mode,                 # "real" or "virtual"
                model_type = args_list[i].model_type,               # "mllm" or "llm", use "llm" as default
                api_config = api_config,                            # api config, include temperature, max tokens, etc
                game_name = args_list[i].game_name,                 # evaluated game
                AP_num = AP_num,                                    # num of Action Point
                bias_type = args_list[i].bias_type,                 # type of evaluated bias type
                bias_feture = bias_feature,                         # feature of evaluated bias
                role_num = args_list[i].role_num,                   # num of roles in role_list
                repeated_num = args_list[i].repeated_num,           # number of repeated tests for a query
                redundancy = args_list[i].redundancy,               # number of redundant query
                batch_size = args_list[i].batch_size,               # number of tasks in one batch, related to LLM API access limitation
                RPM_sleep_time = args_list[i].RPM_sleep_time,       # sleep time after every batch of LLM API qurey, to refresh API RPM
            )
        )

        print("-----------------")
        
        # English data analysis
        en_versus_matrix_2d = convert_to_2d_matrix(record_with_lang["english"])
        en_mean_vector = compute_mean_vector_2d(en_versus_matrix_2d)
        print(f"English - Mean vector: {en_mean_vector[0]:.3f}")
    
        # Calculate MCV and percentages
        en_mcv_value = calculate_trace_mcv(en_versus_matrix_2d)
        en_percentage_simple = mcv_to_percentage_simple(en_mcv_value)
        en_percentage_log = mcv_to_percentage_log(en_mcv_value)
    
        print(f"English - MCV: {en_mcv_value:.3f}")
        print(f"English - Convergence percentage (Method 1): {en_percentage_simple:.1f}%")
        print(f"English - Convergence percentage (Method 2): {en_percentage_log:.1f}%")
        print("-----------------")

        # Chinese data analysis
        ch_versus_matrix_2d = convert_to_2d_matrix(record_with_lang["chinese"])
        ch_mean_vector = compute_mean_vector_2d(ch_versus_matrix_2d)
        print(f"Chinese - Mean vector: {ch_mean_vector[0]:.3f}")
    
        # Calculate MCV and percentages
        ch_mcv_value = calculate_trace_mcv(ch_versus_matrix_2d)
        ch_percentage_simple = mcv_to_percentage_simple(ch_mcv_value)
        ch_percentage_log = mcv_to_percentage_log(ch_mcv_value)
    
        print(f"Chinese - MCV: {ch_mcv_value:.3f}")
        print(f"Chinese - Convergence percentage (Method 1): {ch_percentage_simple:.1f}%")
        print(f"Chinese - Convergence percentage (Method 2): {ch_percentage_log:.1f}%")

        analysis = {
            "model_name": args_list[i].model_name,
            "bias_type": args_list[i].bias_type,
            "test_mode": args_list[i].test_mode,
            "role_num": args_list[i].role_num,
            "game_name": args_list[i].game_name,
            "english": {
                "mean_vector": [round(x, 3) for x in en_mean_vector],
                "mcv_value": round(en_mcv_value, 3),
                "percentage_log": round(en_percentage_log, 3),
                "percentage_simple": round(en_percentage_simple, 3),
            },
            "chinese": {
                "mean_vector": [round(x, 3) for x in ch_mean_vector],
                "mcv_value": round(ch_mcv_value, 3),
                "percentage_log": round(ch_percentage_log, 3),
                "percentage_simple": round(ch_percentage_simple, 3),
            }
        }
        analysis_list.append(analysis)
    
    # Dump analysis result into a json file:
    if "/" in args_list[0].model_name: 
        model_name = args_list[0].model_name.split("/")[-1]
    else:
        model_name = args_list[0].model_name
    json_dump_path = './record/Coo_{}_analysis.json'.format(model_name)
    write_to_json(analysis_list, json_dump_path)

    print("-------Evaluation Finished!-------")