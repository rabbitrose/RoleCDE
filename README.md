
<div align="center"><h2>
<img src="figs/logo.png" width="50px" alt="RoleCDE Framework"> RoleCDE: Benchmarking and Mitigating Role–Alignment Trade-offs in Role-Playing Agents </h2></div>

<p align="center">
    <a href="https://github.com/rabbitrose/RoleCDE/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-Apache--2.0-blue.svg" alt="License"></a>
    <!-- <a href="https://arxiv.org/abs/2601.XXXXX"><img src="https://img.shields.io/badge/Paper-Arxiv-red.svg" alt="Paper"></a> -->
    <a href="https://github.com/rabbitrose/RoleCDE/stargazers"><img src="https://img.shields.io/github/stars/rabbitrose/RoleCDE?style=flat-square" alt="Stars"></a>
    <a href="https://github.com/rabbitrose/RoleCDE/issues"><img src="https://img.shields.io/github/issues/rabbitrose/RoleCDE?style=flat-square" alt="Issues"></a>
</p>

> \[!NOTE\]
> 🌟 Love RoleCDE? Star our project on GitHub to get instant updates and show your support!

## 🖼️ Framework Overview

Role CDE is the first benchmark to evaluate role-playing agents under structured conflicts between role-specific values and alignment-oriented constraints. The framework consists of three modular parts:

<p align="center">
  <img src="figs/framework0102.png" width="950" alt="RoleCDE Framework">
  <br>
  <em>The overall pipeline of RoleCDE.</em>
</p>

1.  **Part 1: Construction of Profile-Scenario Pairs**: Generating 8,000 diverse role profiles by enriching demographic seeds from PersonaHub.
2.  **Part 2: Value and Dilemma Generation**: Synthesizing explicitly conflicting values (Role vs. Alignment) and constructing structured dilemmas across three difficulty levels: Easy, Mid, and Hard.
3.  **Part 3: Evaluation and Mitigation**: Organizing roles into 8 semantic categories and evaluating decision outcomes using an LLM-as-judge.

---

## 📊 Dataset Statistics

The benchmark is constructed at scale to ensure diverse evaluation:

| Scale | Detail | 
| :--- | :--- | 
| **Total Profiles** | ~8,000 unique role profiles | 
| **Total Dilemmas** | ~240,000 structured dilemma instances | 
| **Role Categories** | 8 (Business, Tech, Authority, Sports, etc.)  |
| **Difficulty Levels**| Easy, Mid, Hard | 



---

## :loudspeaker: News
- **[2025/01]** Released the RoleCDE benchmark(version 1).
---

## 🚀 Quick Start

### 3. LLM Configuration

Edit your LLM API URL, token, and model information before evaluation in each  `.py` file.
Example:
```python
API_URL = "xxxxxxxxx" 
API_KEY = "xxxxxxx"
MODEL_NAME = "xxxxxxxxx"
```

- Python 3.10+ (Anaconda recommended)
- All Python dependencies in `requirements.txt`
### 1. install environment
```bash
git clone.....
pip install -r requirements.txt
```
### 2. evaluation from API
```bash
cd evaluation_code
python run_LLM_api.py #Get model decision
python judge_LLM_api.py #Get model results
```
### 3. evaluate local LLM models
```bash
cd evaluation_code
python run_LLM_local.py #Get model decision and results
python general_eval.py #Test model general reasoning ability and role play ability
python NLG_eval.py #Test model in BLEU,ROUGE,Bertscore
```


> \[!Tip\]
> - We have provided some test cases in the  `dataset` (nearly 1k roles) to facilitate researchers' initial evaluation.
> - We have provided some model response data (from the complete dataset) under `example_out` for reference.

## TODOs
<details><summary>Click me to show all TODOs</summary>

- [ ] feat: update the complete dataset (8k roles, 240k dilemmas) to huggingface.
- [ ] feat: update the training data and training model weights to huggingface.
- [ ] feat: add RoleCDE PyPI package.

</details>

<p align="right"><a href="#top">🔝Back to top</a></p>


