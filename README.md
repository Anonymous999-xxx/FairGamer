# Introduction

FairGamer: First benchmark exposing LLMs' decision biases in real-game NPC scenarios, revealing critical balance-breaking effects across languages.

This repository is belong to the conference paper titled "FairGamer: Evaluating Biases in the Application of Large Language Models to Video Games".

<div align="center">
  <img src="img/pipeline.png" alt="FairGamer Benchmark Pipeline" width="800" />
</div>

# How to Use

We recommend you use FairGame evaluation program with the asyncio and the AsyncOpenAI libraries, as this program utilizes asynchronous requests for API testing

## Installation

You also need to make sure your python >= 3.9 and install py repositories in requirements.txt :
```bash
pip install -r requirements.txt
```

## Evaluation

We evaluated the following models: **GPT-4o**, **Grok-3**, **Grok-3-mini**, **DeepSeek-V3**, **Qwen2.5-72B**, **Qwen2.5-7B**, **Llama3.1-70B**, **Llama3.1-8B**

### Eval Example: Serving as Non-Player Characters-Real(SNPC-Real)
Taking Grok-3-mini as an example, execute the following command in the command line:
```bash
cd Serving_as_Non-Player_Characters
python SNPC.py --config "./real_grok-3-mini_config.json"
```
The output decisions of the LLM are extracted and recorded in the json file FairGamer/Serving_as_Non-Player_Characters/record/SNPC_real_grok-3-mini_raw-final.json . After evaluating all 8 models, execute the following command:
```bash
python plot_SNPC-real.py"
```
Then you will obtain the specific D<sub>lstd</sub> and D<sub>cl</sub> results in the command line.

### Other Tasks
To test virtual-type tasks, you need to run the following command:
```bash
cd Serving_as_Non-Player_Characters
python SNPC.py --config "./virtual_grok-3-mini_config.json"
python plot_SNPC-virtual.py"
```

To test other tasks (e.g., ICO-Real), modify the command as follows:
```bash
cd Interacting_as_Competitive_Opponents
python ICO.py --config "./real_grok-3-mini_config.json"
python plot_ICO-real.py"
```

The final test results are shown in the table below:
<div align="center">
  <img src="img/D_lstd_Scores.png" alt="FairGamer Benchmark Results" width="800" />
</div>

## Analysis
### The classification of bias levels based on the crowdsourced questionnaire survey:

We collected evaluations from 30 game players via a questionnaire to assess the level of decision bias in the models. The mean probability of the model's output decisions should ideally be an absolute unbiased point, as an ideal model's decisions should not be influenced by task-irrelevant information (e.g., race, profession, nationality, item origin, etc.). The survey revealed that 83.3\% of players considered a fluctuation of ±5\% around the absolute unbiased point as balanced (no bias); 67.7\% regarded a fluctuation between ±5\% and ±15\% as unbalanced (obvious bias); and deviations exceeding ±15\% were deemed highly unbalanced (severe bias). 

We then converted this bias classification into D<sub>lstd</sub> values to distinguish the bias levels of different models across various tasks, as shown in the following table:

| Task | No Bias | Obvious Bias | Severe Bias |
|---------|-------------|----------------|-------------|
| **SNPC** | [0, 0.050] | (0.050, 0.158] | (0.158, +∞) |
| **ICO** | [0, 0.050]$ | (0.050, 0.160] | (0.160, +∞) |
| **GGS** | [0, 0.208]$ | (0.208, 0.651] | (0.651, +∞) |

Based on this classification and referencing Table~\ref{Table: Decision Log Standard Deviation}, in the SNPC-Real task, the top-performing model, Qwen2.5-72B, exhibited noticeable bias with a D<sub>lstd</sub> score of 0.060, and all models demonstrated either noticeable or severe bias. In SNPC-Virtual, only Qwen2.5-72B showed no bias, achieving a D_{lstd} score of 0.018. For ICO-Real and ICO-Virtual, GPT-4o performed best with D<sub>lstd</sub> scores of 0.093 and 0.092, respectively. Yet all models, including GPT-4o, displayed noticeable or severe bias. In GGS-Real, DeepSeek-V3 had the least bias with a D<sub>lstd</sub> score of 0.440, still indicating noticeable bias. In GGS-Virtual, only DeepSeek-V3 showed no bias, scoring 0.155 in D<sub>lstd</sub>.

### LLMs demonstrate isomorphic social/cultural biases toward both real and virtual world content:

Figure below indicates that the tested LLMs' D<sub>lstd</sub> scores exhibit a positive correlation between Real-type and Virtual-type tasks. This implies that bias is an inherent decision-making characteristic of the model and is largely independent of model parameter size.

<div align="center">
  <img src="img/plot_r_n_v.png" alt="HeatMap of task SNPCV (DeepSeek-V3)" width="500" />
</div>