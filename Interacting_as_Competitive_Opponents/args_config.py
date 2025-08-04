import argparse
import json

def parse_args():
    """Parse and return command-line arguments."""
    parser = argparse.ArgumentParser(description="Configuration Parameters of the FairGamer Evaluation")

    # Model name argument (short: -m, long: --model)
    parser.add_argument(
        "-m_n", "--model_name",
        type=str,
        default="deepseek-chat",
        help="Name of evaluated model."
    )
    parser.add_argument(
        "-m_t", "--model_type",
        type=str,
        default="llm",
        help="Type of evaluated model. 'mllm' or 'llm', use 'llm' as default."
    )
    parser.add_argument(
        "-tm", "--test_mode",
        type=str,
        default="virtual",
        help="Test Mode, you can chose 'real' or 'virtual' (default: 'real')"
    )
    parser.add_argument(
        "-gn_path", "--game_name_path",
        type=str,
        help="path of file that records game names"
    )
    parser.add_argument(
        "-gn_max", "--game_name_max_num",
        type=str,
        help="The max num of game names in evaluation."
    )
    parser.add_argument(
        "-c_path", "--country_path",
        type=str,
        help="Path of flavor file for task GGS-real."
    )
    parser.add_argument(
        "-c_max", "--country_max_num",
        type=int,
        help="max num of evaluated countries."
    )
    parser.add_argument(
        "-eva_lang", "--evaluated_languages",
        type=list,
        default=["english", "chinese"], # ["english", "chinese"] # "arabic" is not supported yet
        help="Evaluated languages."
    )
    parser.add_argument(
        "-max_tokens", "--max_tokens",
        type=int,
        default=4096,
        help="max_tokens for LLM"
    )
    parser.add_argument(
        "-temp", "--temperature",
        type=float,
        default=1.0, # use temperature=1.0 as default for all tasks
        help="temperature for LLM"
    )
    parser.add_argument(
        "-top_p", "--top_p",
        type=float,
        default=0.7, # use top_p=0.7 as default for all tasks
        help="top_p for LLM"
    )
    parser.add_argument(
        "-r", "--repeated_num",
        type=int,
        default=10,
        help="Number of repeated tests for a query."
    )
    parser.add_argument(
        "-redu", "--redundancy",
        type=int,
        default=2,
        help="Number of redundancy query."
    )
    parser.add_argument(
        "-b_s", "--batch_size",
        type=int,
        default=10,
        help="Number of tasks in one batch, related to LLM API access limitation."
    )
    parser.add_argument(
        "-sleep", "--RPM_sleep_time",
        type=int,
        default=5,
        help="Sleep time after every batch of LLM API qurey, to refresh API RPM."
    )
    
    # Add a parameter specifically for specifying configuration files:
    parser.add_argument(
        '--config', 
        type=str, 
        default="./config.json",
        help='Path to config file'
    )
    
    args = parser.parse_args()
    
    # If a configuration file is specified, load parameters from the file:
    if args.config:
        with open(args.config, 'r') as f:
            config = json.load(f)
        
        # Force overriding all parameters with values from the configuration file:
        for key, value in config.items():
            setattr(args, key, value)  # Overwrite directly without checking for empty values
    
    return args

# Test the argument parser if run directly
if __name__ == "__main__":
    args = parse_args()
    print("[Debug] Parsed arguments:")
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")
