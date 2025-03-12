# Solve It Easy: Fine-tuning LLMs (Llama, Qwen, ...) to Solve Vietnamese Math Problems.
<div align="center"> <img src="https://scottlattablog.wordpress.com/wp-content/uploads/2018/02/word_problemsemoji.gif?text=Solve+It+Easy+Project" alt="Solve It Easy Project Image" width="60%" /> </div>

## 📖 Overview
This project focuses on fine-tuning Large Language Models (LLMs) such as **LLaMA**, **Qwen** to improve their ability to solve Vietnamese math problems. The goal is to enhance model performance in understanding problem statements, generating step-by-step solutions, and improving accuracy in mathematical reasoning.

## ✨ Highlights
- Fine-tuning LLMs to improve mathematical understanding in Vietnamese.
- Exploring Different Training Approaches: Supervised fine-tuning (SFT), Reinforcement Learning (RLHF) and GRPO (Group Relative Proxy Optimization)

## 🛠 Tech Stack
- Models: LLaMA, Qwen
- Frameworks: PyTorch, Hugging Face Transformers
- Training Techniques: LoRA, QLoRA, PEFT
- Evaluation Metrics: Accuracy

## 📈 Results
### Baselines

| Model                        | GSM8K-Vi (%) | GSM8K-En (%) |
|------------------------------|-------------|-------------|
| LLaMA-3.2-1B-Instruct        | 7.58        | 8.57        |
| LLaMA-3.2-3B-Instruct        | 24.56       | 20.69       |
| LLaMA-3.1-8B-Instruct        | 34.57       | 33.05       |
| Qwen2.5-3B-Instruct          | 35.03       | 38.67       |
| Qwen2.5-Math-1.5B            | 0.00        | 0.08        |
| Qwen2.5-Math-1.5B-Instruct   | 0.00        | 9.00        |
| AceMath-1.5B-Instruct        | 21.99       | 42.00       |
