# Introduction

FairGamer: First benchmark the first benchmark to evaluate social biases (class, race, age, and nationality) across three interaction patterns: transaction, cooperation, and competition.

This repository is belong to the conference paper titled "FairGamer: Evaluating Social Biases in LLM-Based Video Game NPCs".

Conception of social biases in interactions between NPCs:
<div align="center">
  <img src="img/Conception.png" alt="Conception of social biases in interactions between NPCs" width="800" />
</div>

FairGamer Benchmark Evaluation Pipeline:
<div align="center">
  <img src="img/pipeline_v2.png" alt="FairGamer Benchmark Evaluation Pipeline" width="800" />
</div>

# How to Use

We recommend you use FairGame evaluation program with the asyncio and the AsyncOpenAI libraries, as this program utilizes asynchronous requests for API testing

## Installation

You also need to make sure your python >= 3.9 and install py repositories in requirements.txt :
```bash
pip install -r requirements.txt
```

## Evaluation

We evaluated the following models: **GPT-4.1**, **Grok-4**, **Grok-4-fast**, **DeepSeek-V3.2**, **Qwen2.5-72B**, **Llama3.3-70B**, **Llama3.1-8B**

### Eval: Trasaction (Tr)
Taking DeepSeek-V3.2 as an example, execute the following command in the command line:
```bash
cd FairGamer/Trade
python eval_all_Tr.py --config "./config_dsv3.json"
```
The output decisions of the LLM are extracted as a json, and averaged json outputs will be recorded in a json file in FairGamer/Trade/record/Tr_deepseek-chat_analysis.json . 

After evaluating all 7 models in Trasaction pattern, execute the following command:
```bash
cd FairGamer/Trade
python mcv_Tr.py"
```
Then you will obtain the specific FairMCV results in the command line.

### Eval: Cooperation (Coo)
Like Trasaction, to test Cooperation Pattern, you need to run the following command:
```bash
cd FairGamer/Cooperation
python eval_all_Coo.py --config "./config_dsv3.json"
python mcv_Coo.py"
```

### Eval: Competition (Com)
Like Trasaction, to test Competition Pattern, you need to run the following command:
```bash
cd FairGamer/Competition
python eval_all_Coom.py --config "./config_dsv3.json"
python mcv_Com.py"
```

The final test results are shown in the table below (with English data):
<div align="center">
  <img src="img/FairMCV_Score.png" alt="FairGamer Benchmark Results" width="800" />
</div>

For clarity, an LLM with a FairMCV score above 95% is interpreted as a sufficiently fair model without bias.

## Analysis

We averaged the experimental results separately by interaction modes and bias types, yielding the following outcomes:

<div align="center">
<table>
<tr>
<td align="center"><img src="img/inter_bar_chart_en.png" alt="HeatMap of task SNPCV (DeepSeek-V3)" width="95%" /></td>
<td align="center"><img src="img/bias_bar_chart_en.png" alt="HeatMap of task SNPCV (DeepSeek-V3)" width="95%" /></td>
</tr>
<tr>
<td align="center">(a) Fairness performance across 3 interaction scenarios (English data): Transaction (Tr), Cooperation (Coo), and Competition (Com).</td>
<td align="center">(b) Model fairness performance across 4 social bias types (English data):
Class, Race, Age, and Nationality.</td>
</tr>
</table>
</div>

<div align="center">
<table>
<tr>
<td align="center"><img src="img/inter_bar_chart_zh.png" alt="HeatMap of task SNPCV (DeepSeek-V3)" width="95%" /></td>
<td align="center"><img src="img/bias_bar_chart_zh.png" alt="HeatMap of task SNPCV (DeepSeek-V3)" width="95%" /></td>
</tr>
<tr>
<td align="center">(c) Fairness performance across 3 interaction scenarios (Chinese data): Transaction (Tr), Cooperation (Coo), and Competition (Com).</td>
<td align="center">(d) Model fairness performance across 4 social bias types (Chinese data):
Class, Race, Age, and Nationality.</td>
</tr>
</table>
</div>

Figures (a) and (c) reflect, to some extent, the differences in biases exposed by different models across the three interaction modes. Meanwhile, (b) and (d) illustrate the extent of biases present in these models across specific categories. Note that in the four figures above, higher values indicate greater fairness of the model.

## Design Philosophy of FairMCV:
The fairness metric we propose, FairMCV, attempts to link the fairness of model outputs with convergence. When an LLM handles tasks unrelated to character settings (Role Information), under ideal fairness conditions, its decisions should remain consistent across different characters. In such cases, the model’s decision outputs should be convergent, meaning that higher output consistency indicates greater fairness.