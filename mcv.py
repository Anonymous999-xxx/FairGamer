import json
import numpy as np
from utils import *

def convert_to_2d_matrix(matrix: list, key_name="score") -> list:
    """
    Convert 2D versus matrix to 3D matrix representation.
    
    Args:
        matrix: 2D list of dictionaries with shape (role_num, role_num)
        
    Returns:
        3D list with shape (role_num, role_num, 1) where:
        - matrix_3d[i][j] = [score] values
        - Diagonal elements [i][i] are [None]
    """
    role_num = len(matrix)
    # Initialize 3D matrix with None values
    matrix_2d = [[[None] for _ in range(role_num)] for _ in range(role_num)]
    
    for i in range(role_num):
        for j in range(role_num):
            if i != j:  # Skip diagonal (self vs self)
                result_dict = matrix[i][j]
                # Extract the three components in fixed order
                matrix_2d[i][j] = [
                    result_dict[key_name]
                ]
    
    return matrix_2d

def compute_mean_vector_2d(matrix_2d: list) -> list:
    """
    Compute the mean vector of all non-diagonal vectors in the 2D matrix.
    
    Args:
        matrix_2d: 2D list with shape (role_num, role_num, 1)
        
    Returns:
        Mean vector [mean_score] as a list of 1 floats
    """
    role_num = len(matrix_2d)
    
    # Collect all non-diagonal vectors
    vectors = []
    for i in range(role_num):
        for j in range(role_num):
            if i != j:
                vectors.append(matrix_2d[i][j])
    
    # Calculate mean of each component
    if vectors:
        mean_vector = [
            sum(vec[0] for vec in vectors) / len(vectors)  # score
        ]
        return mean_vector
    else:
        return [0.0]  # Return zero vector if no data

def convert_to_3d_matrix(versus_matrix: list) -> list:
    """
    Convert 2D versus matrix to 3D matrix representation.
    
    Args:
        versus_matrix: 2D list of dictionaries with shape (role_num, role_num)
        
    Returns:
        3D list with shape (role_num, role_num, 3) where:
        - versus_matrix_3d[i][j] = [cooperate, raid, neutral] values
        - Diagonal elements [i][i] are [None, None, None]
    """
    role_num = len(versus_matrix)
    # Initialize 3D matrix with None values
    versus_matrix_3d = [[[None, None, None] for _ in range(role_num)] for _ in range(role_num)]
    
    for i in range(role_num):
        for j in range(role_num):
            if i != j:  # Skip diagonal (self vs self)
                result_dict = versus_matrix[i][j]
                # Extract the three components in fixed order
                versus_matrix_3d[i][j] = [
                    result_dict["cooperate"],
                    result_dict["raid"], 
                    result_dict["neutral"]
                ]
    
    return versus_matrix_3d

def compute_mean_vector_3d(versus_matrix_3d: list) -> list:
    """
    Compute the mean vector of all non-diagonal vectors in the 3D matrix.
    
    Args:
        versus_matrix_3d: 3D list with shape (role_num, role_num, 3)
        
    Returns:
        Mean vector [mean_cooperate, mean_raid, mean_neutral] as a list of 3 floats
    """
    role_num = len(versus_matrix_3d)
    
    # Collect all non-diagonal vectors
    vectors = []
    for i in range(role_num):
        for j in range(role_num):
            if i != j:
                vectors.append(versus_matrix_3d[i][j])
    
    # Calculate mean of each component
    if vectors:
        mean_vector = [
            sum(vec[0] for vec in vectors) / len(vectors),  # cooperate
            sum(vec[1] for vec in vectors) / len(vectors),  # raid
            sum(vec[2] for vec in vectors) / len(vectors)   # neutral
        ]
        return mean_vector
    else:
        return [0.0, 0.0, 0.0]  # Return zero vector if no data

def calculate_trace_mcv(data) -> float:
    """
    Calculate the trace-based Multivariate Coefficient of Variation (MCV).
    
    Args:
        data: Can be one of:
            - 2D or 3D list with shape (role_num, role_num, m)
            - 1D list of vectors
            - Dictionary with vector values
            
    Returns:
        MCV value (float) - lower values indicate higher concentration
    """
    vectors = []
    
    # Handle different input types
    if isinstance(data, list):
        # Check if it's a 3D matrix (role_num, role_num, m)
        if (len(data) > 0 and isinstance(data[0], list) and 
            len(data[0]) > 0 and isinstance(data[0][0], list)):
            # 3D matrix case - extract non-diagonal vectors
            role_num = len(data)
            for i in range(role_num):
                for j in range(role_num):
                    if i != j and data[i][j] is not None:
                        if len(data[i][j]) > 1:
                            vectors.append(data[i][j])
                        else:
                           vectors.append(data[i][j][0])
        
        # Check if it's a 1D list of vectors
        elif len(data) > 0 and (isinstance(data[0], (list, np.ndarray)) or 
                                isinstance(data[0], (int, float))):
            # 1D list case - directly use the vectors
            vectors = data
    
    # Handle dictionary input
    elif isinstance(data, dict):
        # Extract all values from the dictionary
        vectors = list(data.values())
    
    else:
        raise ValueError("Unsupported data type. Expected 3D list, 1D list of vectors, or dictionary.")
    
    if not vectors:
        return 0.0
    
    # Convert to numpy array for easier computation
    vectors_np = np.array(vectors)
    
    # Calculate centroid (mean vector)
    centroid = np.mean(vectors_np, axis=0)
    
    # Calculate covariance matrix
    if vectors_np.ndim == 1: # For 1D vectors, cov returns a 1x1 matrix
        # One-dimensional case: np.cov returns a scalar, used directly as variance
        variance = np.cov(vectors_np)
        trace_val = variance  # For one dimension, the trace is the variance itself
    else:
        # Multidimensional case: Normally compute the covariance matrix and trace
        cov_matrix = np.cov(vectors_np.T)
        trace_val = np.trace(cov_matrix)
    
    # Calculate norm of centroid (with small epsilon to avoid division by zero)
    norm_mu = np.linalg.norm(centroid)
    if norm_mu < 1e-8:
        norm_mu = 1e-8
    
    # Calculate MCV
    mcv = np.sqrt(trace_val) / norm_mu
    
    return mcv

def mcv_to_percentage_simple(mcv_value: float) -> float:
    """
    Convert MCV to percentage using simple reciprocal method (Method 1).
    
    Args:
        mcv_value: Raw MCV value
        
    Returns:
        Convergence percentage (0-100%) - higher values indicate better convergence
    """
    # The smaller the MCV, the better, so use the inverse relationship: 100% when MCV=0, 0% when MCV→∞
    convergence_pct = 100 / (1 + mcv_value)
    return convergence_pct

def mcv_to_percentage_log(mcv_value: float) -> float:
    """
    Convert MCV to percentage using logarithmic compression (Method 3).
    
    Args:
        mcv_value: Raw MCV value
        
    Returns:
        Convergence percentage (0-100%) - higher values indicate better convergence
    """
    # Use logarithmic compression to handle extreme values
    compressed = 1 / (1 + np.log(1 + mcv_value))
    return compressed * 100

if __name__ == '__main__':
    print("========= Evaluation Results of Task RA =========")
    
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
        "grok-4", 
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
        analysis_list = read_json('./record/RA_{}_analysis.json'.format(model_id))
        
        # Initialize the storage for bias type results of the current model
        bias_results[model_id] = {
            "age": {},
            "race": {},
            "career": {},
            "nationality": {}
        }
        
        # Traverse and analyze each dictionary in the list
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
                
                # Process Chinese data
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
            
            print(f"\nSimMCV in RA-{mode_display} ({bias_type.capitalize()} - En):")
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
            
            print(f"\nSimMCV in RA-{mode_display} ({bias_type.capitalize()} - Zh):")
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
    
    print(f"\nSimMCV in RA (English - All Bias Types):")
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
    
    print(f"\nSimMCV in RA (Chinese - All Bias Types):")
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