import json
import numpy as np
from utils import *

import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
from mcv import *

if __name__ == '__main__':
    print("========= Evaluation Results of Task Com =========")
    
    model_name = [
        "GPT-4.1",
        "Grok-4",
        "Grok-4-fast",
        "DeepSeek-V3.2", 
        "Qwen2.5-72B",
        "Llama3.3-70B", 
        "Llama3.1-8B"
    ]
    model_list = [
        "gpt-4.1",
        "grok-4-0709", 
        "grok-4-fast-non-reasoning",
        "deepseek-chat", 
        "qwen2.5-72b-instruct",
        "Meta-Llama-3.3-70B-Instruct",
        "Meta-Llama-3.1-8B-Instruct"
    ]
    
    # Initialize the dictionary to store results
    bias_results = {}
    
    # Traverse all models
    for i, model_id in enumerate(model_list):
        # Read analysis result file
        analysis_list = read_json('./record/Com_{}_analysis.json'.format(model_id))
        
        # Initialize the bias type result storage for the current model
        bias_results[model_id] = {
            "age": {},
            "race": {},
            "career": {},
            "nationality": {}
        }
        
        # Traverse each dictionary in the analysis list
        for analysis_item in analysis_list:
            bias_type = analysis_item["bias_type"]
            test_mode = analysis_item["test_mode"]
            
            # Check if the bias type is within the predefined types
            if bias_type in ["age", "race", "career", "nationality"]:
                # Determine storage key based on test mode
                storage_key = test_mode if test_mode in ["real", "virtual"] else None
                
                # Ensure the storage key exists
                if storage_key not in bias_results[model_id][bias_type]:
                    bias_results[model_id][bias_type][storage_key] = {"en_mcv": [], "en_percentage_log": [], "ch_mcv": [], "ch_percentage_log": []}
                
                # Process English data
                en_mcv_value = analysis_item["english"]["mcv_value"]
                en_percentage_log = mcv_to_percentage_log(en_mcv_value)
                bias_results[model_id][bias_type][storage_key]["en_mcv"].append(en_mcv_value)
                bias_results[model_id][bias_type][storage_key]["en_percentage_log"].append(en_percentage_log)
                
                # Processing Chinese data
                ch_mcv_value = analysis_item["chinese"]["mcv_value"]
                ch_percentage_log = mcv_to_percentage_log(ch_mcv_value)
                bias_results[model_id][bias_type][storage_key]["ch_mcv"].append(ch_mcv_value)
                bias_results[model_id][bias_type][storage_key]["ch_percentage_log"].append(ch_percentage_log)
    
    # Calculate the average for each model, each bias type, and test mode
    model_avg_results = {}
    for model_id in model_list:
        model_avg_results[model_id] = {}
        for bias_type in ["age", "race", "career", "nationality"]:
            model_avg_results[model_id][bias_type] = {}
            
            # Get all test patterns under this bias type
            test_modes = list(bias_results[model_id][bias_type].keys())
            
            for test_mode in test_modes:
                # Calculate the average of English data
                en_mcv_list = bias_results[model_id][bias_type][test_mode]["en_mcv"]
                en_percentage_log_list = bias_results[model_id][bias_type][test_mode]["en_percentage_log"]
                
                en_mcv_avg = sum(en_mcv_list) / len(en_mcv_list) if en_mcv_list else 0
                en_percentage_log_avg = sum(en_percentage_log_list) / len(en_percentage_log_list) if en_percentage_log_list else 0
                
                # Calculate the average of Chinese data
                ch_mcv_list = bias_results[model_id][bias_type][test_mode]["ch_mcv"]
                ch_percentage_log_list = bias_results[model_id][bias_type][test_mode]["ch_percentage_log"]
                
                ch_mcv_avg = sum(ch_mcv_list) / len(ch_mcv_list) if ch_mcv_list else 0
                ch_percentage_log_avg = sum(ch_percentage_log_list) / len(ch_percentage_log_list) if ch_percentage_log_list else 0
                
                model_avg_results[model_id][bias_type][test_mode] = {
                    "en_mcv_avg": en_mcv_avg,
                    "en_percentage_log_avg": en_percentage_log_avg,
                    "ch_mcv_avg": ch_mcv_avg,
                    "ch_percentage_log_avg": ch_percentage_log_avg
                }
    
    # Output the result table for each bias type and test mode combination
    bias_types = ["age", "race", "career", "nationality"]
    
    for bias_type in bias_types:
        # Get all test patterns under this bias type
        test_modes = list(model_avg_results[model_list[0]][bias_type].keys()) if model_list else []
        
        for test_mode in test_modes:
            # Build table title
            mode_display = test_mode if test_mode is not None else "None"
            
            print(f"\nSimMCV in Com-{mode_display} ({bias_type.capitalize()} - En):")
            print("------------------------------------------------------------------------------")
            print("| Model Name | MCV Value | SimMCV |")
            for i, model_id in enumerate(model_list):
                avg_data = model_avg_results[model_id][bias_type][test_mode]
                print("| {} | {:.3f} | {:.1f} |".format(
                    model_name[i],
                    avg_data["en_mcv_avg"],
                    avg_data["en_percentage_log_avg"]
                ))
            print("------------------------------------------------------------------------------")
            
            print(f"\nSimMCV in Com-{mode_display} ({bias_type.capitalize()} - Zh):")
            print("------------------------------------------------------------------------------")
            print("| Model Name | MCV Value | SimMCV |")
            for i, model_id in enumerate(model_list):
                avg_data = model_avg_results[model_id][bias_type][test_mode]
                print("| {} | {:.3f} | {:.1f} |".format(
                    model_name[i],
                    avg_data["ch_mcv_avg"],
                    avg_data["ch_percentage_log_avg"]
                ))
            print("------------------------------------------------------------------------------")
    
    # Print full table of en and zh:
    
    print(f"\nSimMCV in Com (English - All Bias Types):")
    print("----------------------------------------------------------------------------------------")
    print("| Model Name     | Career | Race | Age | Nationality |")
    print("----------------------------------------------------------------------------------------")
    for i, model_id in enumerate(model_list):
        career_data = model_avg_results[model_id]["career"]
        race_data = model_avg_results[model_id]["race"] 
        age_data = model_avg_results[model_id]["age"]
        nationality_data = model_avg_results[model_id]["nationality"]
        
        # Career: Calculate the average of real and virtual
        career_values = []
        if "real" in career_data:
            career_values.append(career_data["real"]["en_percentage_log_avg"])
        if "virtual" in career_data:
            career_values.append(career_data["virtual"]["en_percentage_log_avg"])
        career_avg = sum(career_values) / len(career_values)
        
        # Race: Calculate the average of real and virtual
        race_values = []
        if "real" in race_data:
            race_values.append(race_data["real"]["en_percentage_log_avg"])
        if "virtual" in race_data:
            race_values.append(race_data["virtual"]["en_percentage_log_avg"])
        race_avg = sum(race_values) / len(race_values)
        
        # Age: Direct value
        age_avg = age_data[None]["en_percentage_log_avg"]
        
        # Nationality: Calculate the average of real and virtual
        nationality_values = []
        if "real" in nationality_data:
            nationality_values.append(nationality_data["real"]["en_percentage_log_avg"])
        if "virtual" in nationality_data:
            nationality_values.append(nationality_data["virtual"]["en_percentage_log_avg"])
        nationality_avg = sum(nationality_values) / len(nationality_values)
        
        print("| {:14} | {:6.1f} | {:4.1f} | {:3.1f} | {:11.1f} |".format(
            model_name[i],
            career_avg,
            race_avg, 
            age_avg,
            nationality_avg
        ))
    print("----------------------------------------------------------------------------------------")
    
    print(f"\nSimMCV in Com (Chinese - All Bias Types):")
    print("----------------------------------------------------------------------------------------")
    print("| Model Name     | Career | Race | Age | Nationality |")
    print("----------------------------------------------------------------------------------------")
    for i, model_id in enumerate(model_list):
        career_data = model_avg_results[model_id]["career"]
        race_data = model_avg_results[model_id]["race"]
        age_data = model_avg_results[model_id]["age"]
        nationality_data = model_avg_results[model_id]["nationality"]
        
        # Career: Calculate the average of real and virtual
        career_values = []
        if "real" in career_data:
            career_values.append(career_data["real"]["ch_percentage_log_avg"])
        if "virtual" in career_data:
            career_values.append(career_data["virtual"]["ch_percentage_log_avg"])
        career_avg = sum(career_values) / len(career_values)
        
        # Race: Calculate the average of real and virtual
        race_values = []
        if "real" in race_data:
            race_values.append(race_data["real"]["ch_percentage_log_avg"])
        if "virtual" in race_data:
            race_values.append(race_data["virtual"]["ch_percentage_log_avg"])
        race_avg = sum(race_values) / len(race_values)
        
        # Age: Direct value
        age_avg = age_data[None]["ch_percentage_log_avg"]
        
        # Nationality: Calculate the average of real and virtual
        nationality_values = []
        if "real" in nationality_data:
            nationality_values.append(nationality_data["real"]["ch_percentage_log_avg"])
        if "virtual" in nationality_data:
            nationality_values.append(nationality_data["virtual"]["ch_percentage_log_avg"])
        nationality_avg = sum(nationality_values) / len(nationality_values)
        
        print("| {:14} | {:6.1f} | {:4.1f} | {:3.1f} | {:11.1f} |".format(
            model_name[i],
            career_avg,
            race_avg,
            age_avg,
            nationality_avg
        ))
    print("----------------------------------------------------------------------------------------")
