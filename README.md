# Solve It Easy: Fine-tuning LLMs (Llama, Qwen, ...) to Solve Vietnamese Math Problems.
<div align="center"> <img src="https://scottlattablog.wordpress.com/wp-content/uploads/2018/02/word_problemsemoji.gif?text=Solve+It+Easy+Project" alt="Solve It Easy Project Image" width="60%" /> </div>

## 📖 Overview
This project focuses on fine-tuning Large Language Models (LLMs) such as **LLaMA**, **Qwen** to improve their ability to solve Vietnamese math problems. The goal is to enhance model performance in understanding problem statements, generating step-by-step solutions, and improving accuracy in mathematical reasoning.

## ✨ Highlights
- **Supervised Fine-tuning LLMs**: Enhancing performance for solving Vietnamese math problems using models like T5, LLaMA, Qwen.
- **Training Optimization**: Applying PEFT techniques like LoRA, QLoRA for efficient fine-tuning and faster processing.
- **Post-training LLMs**: Reinforcement Learning (RLHF) and GRPO (Group Relative Proxy Optimization)
- **Model Comparison**: Evaluating LLaMA, Qwen, and other models to determine the most effective for solving math problems in Vietnamese.

## 📌 Prerequisites
- Python >= 3.8
- CUDA-enabled GPU (Optional but recommended)
- Libraries: PyTorch, Hugging Face Transformers, and other dependencies listed in `requirements.txt`.

## 📈 Results
### Baselines

## 🏗 Data Preparation
This project uses a Vietnamese-translated version of the **GSM8K** (Grade School Math World Problem 8.5K) dataset, which consists of 8,000 math word problems. These problems range from basic arithmetic to more complex logical reasoning and algebraic problems. The dataset is designed to test the ability of models to understand and solve math problems in Vietnamese.

### Data Format
The dataset is provided in a structured **JSON** format. Each problem in the dataset is represented by a dictionary containing two fields:
- **Question**: The math word problem written in Vietnamese.
- **Answer**: The correct solution to the problem.

Example of a data entry:
```json
{
      "question": "Mark has a garden with flowers. He planted plants of three different colors in it. Ten of them are yellow, and there are 80% more of those in purple. There are only 25% as many green flowers as there are yellow and purple flowers. How many flowers does Mark have in his garden?",
      "answer": "There are 80/100 * 10 = <<80/100*10=8>>8 more purple flowers than yellow flowers. So in Mark's garden, there are 10 + 8 = <<10+8=18>>18 purple flowers.\nPurple and yellow flowers sum up to 10 + 18 = <<10+18=28>>28 flowers.\nThat means in Mark's garden there are 25/100 * 28 = <<25/100*28=7>>7 green flowers.\nSo in total Mark has 28 + 7 = <<28+7=35>>35 plants in his garden.\n#### 35"
}
```
### Preprocessing
To prepare the dataset for model training, we need to extract the explanation and the final numeric answer from the provided raw `answer` field.  
- **Extract Explanation**: Keep the full explanation, including any necessary intermediate calculations.  
- **Extract Final Answer**: Capture the final numeric answer, which is usually located at the end of the answer string.  

Example: Given:

```json
{
  "question": "Mark has a garden with flowers. He planted plants of three different colors in it. Ten of them are yellow, and there are 80% more of those in purple. There are only 25% as many green flowers as there are yellow and purple flowers. How many flowers does Mark have in his garden?",
  "answer": "There are 80/100 * 10 = 8 more purple flowers than yellow flowers. So in Mark's garden, there are 10 + 8 = 18 purple flowers. Purple and yellow flowers sum up to 10 + 18 = 28 flowers. That means in Mark's garden there are 25/100 * 28 = 7 green flowers. So in total Mark has 28 + 7 = 35 plants in his garden. #### 35"
}
```
After preprocessing:

```json
{
  "question": "Mark has a garden with flowers. He planted plants of three different colors in it. Ten of them are yellow, and there are 80% more of those in purple. There are only 25% as many green flowers as there are yellow and purple flowers. How many flowers does Mark have in his garden?",
  "solution": "There are 80/100 * 10 = 8 more purple flowers than yellow flowers. So in Mark's garden, there are 10 + 8 = 18 purple flowers. Purple and yellow flowers sum up to 10 + 18 = 28 flowers. That means in Mark's garden there are 25/100 * 28 = 7 green flowers. So in total Mark has 28 + 7 = 35 plants in his garden.",
  "answer": "35"
}
```

### Prompt Construction
Before using the dataset for training or evaluation, we need to format the math problem into a input that is suitable for Large Language Models (LLMs). This involves building a prompt for the LLM to understand the input problem clearly.

#### Prompt Template
To maintain consistency in how the problems are presented to the model, use the following prompt template for constructing problem statements:
```
You are a math expert.

Solve the following problem carefully and accurately.
Your response must include two sections:
- Reason: Provide a step-by-step explanation of how to solve the problem.
- Answer: Write only the final numerical answer without any additional text, symbols, or units.

Format your response exactly as follows:
Solution:
Reason:
<Step-by-step explanation>
Answer:
<Final numerical result>

Here is the problem:
{problem}

Now, please solve the problem and provide your response below.
{solution}
```

#### Example of Preprocessed Data
After preprocessing, each entry in the dataset will have the same structure but will include the formatted prompt for LLM input. Here's an example of what a preprocessed entry might look like:
```json
{
    "index": 5434,
    "answer": "274",
    "problem": "Kylie và Kayla hái táo cùng nhau và mang về nhà tổng cộng 340 quả táo. Nếu Kayla hái được nhiều hơn 4 lần số quả táo mà Kylie hái thì 10 quả táo, hỏi Kayla hái được bao nhiêu quả táo?",
    "solution": "Solution:\nReason:\nKylie đã hái được 66 quả táo. Kayla đã hái được 274 quả táo.\nAnswer:\n274"
    "text": "You are a math expert.\n\nSolve the following problem carefully and accurately...Vì vậy, Marc đã chi tiêu tổng cộng $100 + $50 + $10 = $160.\nAnswer:\n160"
}
```
This preprocessed data, including the prompt, is now ready for tokenization and model training.

### 🔧 Configuration Setup
To perform experiments, we use a `.yaml` configuration file. This file serves as the central configuration for the entire workflow of project. It is used to set various parameters for all stages, including data preparation, training, evaluation, and inference. 

#### Parameters
Below is an example of the key sections of the configuration:

<details>
  <summary><code>exp_manager</code></summary>

  ```yaml
  # Controls experiment-level settings, such as project name, experiment tracking, and logging.
  # Some key parameters:
  - `prj_name`: Name of the overall project (e.g., `"llm_math_problem_solve"`).
  - `exp_name`: Name of the specific experiment (e.g., `"llama-3.1-8b-instruct__gsm8k_vi"`).
  - `wandb`: Configuration for Weights & Biases logging.
    - `project`: W&B project name.
    - `log_artifact`: Whether to log artifacts (`false` by default).
  ```
</details>

<details>
  <summary><code>prepare_model</code></summary>

```yaml
  # Configures the model and its hyperparameters.  
  # Some key parameters:
  - `model_type`: Type of model (e.g., `causal_lm`).
  - `pretrained_model_name_or_path`: Path to the pretrained model (e.g., `meta-llama/Llama-3.1-8B-Instruct`).
  - `load_in_4bit`: Whether to load the model in 4-bit precision (e.g., `false`).
```
</details>

<details>
  <summary><code>prepare_data</code></summary>

```yaml
  # Defines the dataset settings and preprocessing options.  
  # Some key parameters:
  - `dataset_name_or_path`: Path to the dataset (e.g., `'namfam/gsm8k-vietnamese'`).
  - `do_tokenize`: Whether to tokenize the data (e.g., `true`).
  - `save_prepared`: Whether to save the preprocessed data (e.g., `true`).
```
</details>

<details>
  <summary><code>tokenizer</code></summary>

```yaml
  # Handles tokenizer-specific settings.  
  # Some key parameters:
  - `new_pad_token`: Define a new padding token if needed (e.g., `null`).
  - `do_add_special_tokens`: Whether to add special tokens (e.g., `false`).
  - `max_length`: Maximum length for tokenization (e.g., `512`).
```
</details>

<details>
  <summary><code>prompt</code></summary>
      
```yaml
  # Specifies how the input prompt is formatted.  
  # Some key parameters:**
  - `intro_text`: Introductory message (e.g., `"You are a math expert."`).
  - `instruction_text`: Instruction for the model (e.g., `"Solve the following problem carefully and accurately."`).
  - `response_format_guide_text`: Example format for responses (e.g., `"Solution:\nReason:\n<Step-by-step explanation>\nAnswer:\n<Final numerical result>"`).\
```
</details>

<details>
  <summary><code>train</code></summary>

  ```yaml
  # Controls the training parameters.  
  # Some key parameters:
  - `use_peft`: Whether to use Parameter-Efficient Fine-Tuning (e.g., `true`).
  - `train_n_samples`: Number of samples for training (e.g., `-1` for all data).
  - `train_args`: HuggingFace TrainingArguments for training, including learning rate, batch size, etc.
  ```
</details>

<details>
  <summary><code>evaluate</code></summary>

  ```yaml
  # Defines the settings for evaluating the model.  
  # Some key parameters:
  - `eval_dataset_name`: Name of the dataset to evaluate on (e.g., `'test'`).
  - `batch_size`: Batch size for evaluation (e.g., `128`).
  - `prediction_filename`: File to save model predictions (e.g., `'test_predictions.txt'`).
  ```
</details>

<details>
  <summary><code>generate</code></summary>

  ```yaml
  # Configures the generation settings for inference.  
  # Some key parameters:**
  - `seed`: Random seed for generating results (e.g., `202502`).
  - `max_new_tokens`: Maximum number of tokens to generate (e.g., `512`).
  - `temperature`: Sampling temperature for randomness (e.g., `1.0`).
  ```
</details>


#### How to Use the Configuration File
Create or modify the `.yaml` configuration file as per your experiment needs.
Run the data prep, training, evaluation, inference script with the configuration file as an argument:
```bash
python prepare_data.py --config_path configs/<config_file>.yaml
# python finetune.py --config_path configs/<config_file>.yaml
# python eval.py --config_path configs/<config_file>.yaml
# python generate.py --config_path configs/<config_file>.yaml
```
The script will automatically read the configuration, load the data, and initiate the process for the corresponding stage (data preparation, training, evaluation, or inference) based on the parameters set in the .yaml file.

## 🏋️ Training/Finetuning
```bash
python finetune.py --config_path configs/....yaml
```
Support finetuning for the `Causal Language Model` (e.g., BART, T5) and `Seq2Seq Language Model` (e.g., Llama, Qwen, Gemma, Mistral).

## 📊 Evaluation
Evaluate the fine-tuned model with:
```bash
python eval.py --config_path configs/....yaml
```
Example output:
```bash
Accuracy: 90.1%
```

## 🚀 Inference
Generate solution for new math word problem:
```bash
python generate.py \
  --config_path configs/....yaml \
  --input "Một chiếc áo choàng cần 2 cuộn sợi màu xanh và một nửa số đó là sợi màu trắng. Tổng cộng cần bao nhiêu cuộn sợi?"
```

Example output:
```bash
Solution:
Reason:
Cần 2/2=1 cuộn sợi màu trắng.
Vì vậy tổng số lượng vải là 2+1=3 cuộn vải.
Answer:
3
```

## 🤝 Contributions
Contributions are welcome! Please submit a pull request or open an issue.

## 📜 License
This project is licensed under the [Your License] License. See the LICENSE file for details.
