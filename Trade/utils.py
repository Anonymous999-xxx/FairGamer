import os
import json
import asyncio
import time
import random
import matplotlib.pyplot as plt
from collections import Counter

def read_txt(file_name: str) -> str:
    """Read .txt file"""
    # Read file content
    current_dir = os.getcwd() # Get the current directory
    file_path = os.path.join(current_dir, file_name) # Specify the file path
    with open(file_path, 'r', encoding='utf-8') as file:
        txt_file = file.read()
    return txt_file

def read_json(file_name: str) -> dict:
    """Read JSON file and return a dictionary"""
    with open(file_name, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

def write_to_json(data, file_path, ensure_ascii=False, indent=4):
    try:
        with open(file_path, 'w', encoding='utf-8') as json_file:
            json.dump(data, json_file, ensure_ascii=ensure_ascii, indent=indent)
        print(f"Successfully written to file: {file_path}")
    except Exception as e:
        print(f"An error occurred while writing to the file: {e}")

class TokenBucket:
    def __init__(self, rate_limit: int, interval: int = 60):
        """
        :param rate_limit: The request limit per minute (e.g., 1500)
        :param interval: The time interval (default is 60 seconds)
        """
        self.rate_limit = rate_limit
        self.interval = interval
        self.tokens = rate_limit  # Initialize the token bucket as full
        self.last_refill_time = time.time()

    def _refill_tokens(self):
        """Supplementary Tokens"""
        now = time.time()
        time_elapsed = now - self.last_refill_time
        new_tokens = (time_elapsed / self.interval) * self.rate_limit
        if new_tokens > 0:
            self.tokens = min(self.rate_limit, self.tokens + new_tokens)
            self.last_refill_time = now

    async def acquire(self):
        """Get a token, wait if there is no token"""
        while True:
            self._refill_tokens()
            if self.tokens >= 1:
                self.tokens -= 1
                break
            else:
                # If there is no token, wait for a period of time and try again
                await asyncio.sleep(0.1)

class RetryLimitExceededError(Exception):
    """Custom exception to indicate that the number of retries is too high."""
    def __init__(self, message="Retry limit exceeded!"):
        self.message = message
        super().__init__(self.message)
        
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

def select_elements_from_the_top(elements: list, truncation: int) -> list:
    """
    Select elements for each sublist, with truncation from the beginning to the end.
    
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

def select_elements_with_truncation(elements: list, truncation: int) -> list:
    """
    Select elements for each sublist, with truncation from the beginning to the end.
    
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

def draw_distribution(record):
    # Count the occurrence of each profession
    counter = Counter(record)

    # Extract occupation and corresponding occurrence count
    labels = list(counter.keys())
    counts = list(counter.values())

    # Draw bar chart
    colors = ['blue', 'green', 'red']
    bars = plt.bar(labels, counts, color=colors)

    # Add legend
    plt.legend(bars, labels)

    plt.xlabel('职业')
    plt.ylabel('出现次数')
    plt.title('职业出现次数统计')

    # Display image and wait for key press
    plt.show(block=False)  # Non-blocking display of images
    plt.waitforbuttonpress()  # Wait for user key press

    # Close image window
    plt.close()