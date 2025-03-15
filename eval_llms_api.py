from litellm import batch_completion


gen_answer_prompt_template = """
You are a math expert in Vietnamese.
---
Solve the following problem carefully and accurately.
Your response must include two sections:
- Reason: Provide a step-by-step explanation of how to solve the problem.
- Answer: Write only the final numerical answer without any additional text, symbols, or units.
---
Format your response exactly as follows:
Solution:
Reason:
<Step-by-step explanation>
Answer:
<Final numerical result>
---
Here is the problem:
{problem}
---
Now, please solve the problem and provide your response below.
Solution:
"""


def generate_prompts(batch):
    prompts = [gen_answer_prompt_template.format(problem=item['problem']) for item in batch]
    return prompts


def generate_prompts(batch):
    """
    Generate prompts for a batch of problems.

    Args:
        batch (dict): A dictionary of tensors.

    Returns:
        list: A list of prompt strings.
    """
    prompts = [gen_answer_prompt_template.format(problem=item['problem']) for item in batch]
    return prompts


def get_responses(prompts):
    """
    Get responses from LiteLLM for a batch of prompts.

    Args:
        prompts (list): A list of prompt strings.

    Returns:
        list: A list of responses from the model.
    """
    messages = [[{"role": "user", "content": prompt}] for prompt in prompts]
    responses = batch_completion(model="gemini/gemini-2.0-flash", messages=messages)
    # response_texts = [response['content'] for response in responses]
    response_texts = [response.choices[0].message['content'] for response in responses]
    return response_texts


from datasets import load_dataset
from torch.utils.data import DataLoader
dataset = load_dataset('namfam/gsm8k-english')

test_ds = dataset['test']
# test_ds = test_ds.with_format("torch")

test_loader = DataLoader(test_ds, batch_size=eval_args.batch_size, shuffle=False)


import evaluate
accuracy_metric = evaluate.load("accuracy")


import re

def extract_numeric_answer(response):
    """
    Extract the numeric answer from the response text.

    Args:
        response (str): The response text.

    Returns:
        str: The extracted numeric answer.
    """
    # Use regular expression to find the answer
    match = re.search(r'Answer:\s*(\d+)', response)
    if match:
        return match.group(1)
    else:
        return None

# Example usage
response = """
Solution:
Reason:
Janet có 16 quả trứng mỗi ngày. Cô ấy ăn 3 quả và dùng 4 quả để làm bánh muffin. Tổng số trứng cô ấy dùng là 3 + 4 = 7 quả. Số trứng còn lại để bán là 16 - 7 = 9 quả. Cô ấy bán mỗi quả trứng với giá $2, vì vậy số tiền cô ấy kiếm được mỗi ngày là 9 quả * $2/quả = $18.

Answer:
18
"""

numeric_answer = extract_numeric_answer(response)
# print(numeric_answer)  # Output: 18


def eval(model_id, test_loader, result_file_types, result_filename, break_step=3):
    accuracy_metric = evaluate.load("accuracy")
    
    # Create a list to store all evaluation results before writing to the file
    results = []

    for step, batch in tqdm(enumerate(test_loader)):
        if step == break_step:  # You can adjust this for testing purposes or remove it to process the entire dataset
            break

        # Get the batch's responses
        ids = batch["index"]
        prompts = generate_prompts(batch)
        responses = get_responses(prompts, model_id)  # Get responses for the batch

        # Extract predicted answers for the batch in one go
        pred_answers = [extract_numeric_answer(response) for response in responses]

        # Process each item in the batch
        for problem, prompt, response, true_solution, true_answer, pred_answer in zip(batch['problem'], prompts, responses, batch['solution'], batch['answer'], pred_answers):
            # Dynamically build the result entry dictionary
            result_entry = {
                'Id': ids,
                'Problem': problem,
                'Prompt': prompt,
                'Response': response,
                'Expected Solution': true_solution,
                'Expected Answer': true_answer,
                'Predicted Answer': pred_answer
            }

            # Store the result entry for later writing
            results.append(result_entry)

            # Add the batch to the accuracy metric
            accuracy_metric.add_batch(
                predictions=[pred_answer],
                references=[true_answer]
            )

    # # Save results based on the selected file types
    # for file_type in result_file_types:
    #     if file_type == 'txt':
    #         write_to_txt(f, results)

    #     elif file_type == 'csv':
    #         fieldnames = results[0].keys()  # Use the keys of the first entry as the CSV column headers
    #         write_to_csv(result_filename + '.csv', results, fieldnames)

    #     elif file_type == 'json':
    #         write_to_json(result_filename + '.json', results)

    #     else:
    #         raise ValueError("Unsupported file type. Please choose 'txt', 'csv', or 'json'.")

    # Return the computed accuracy
    accuracy_result = accuracy_metric.compute()
    return accuracy_result, results
def get_responses(prompts, model_id):
    """
    Get responses from LiteLLM for a batch of prompts.

    Args:
        prompts (list): A list of prompt strings.
        model_id (str): The model ID to use for generating responses.

    Returns:
        list: A list of responses from the model.
    """
    messages = [[{"role": "user", "content": prompt}] for prompt in prompts]
    responses = batch_completion(model=model_id, messages=messages)
    response_texts = [response.choices[0].message['content'] for response in responses]
    return response_texts


# model_id = 'gemini/gemini-1.5-flash-8b'
model_id = 'groq/qwen-qwq-32b'
model_id = 'huggingface/Llama-3.2-1B-Instruct'
test_loader = DataLoader(test_ds, batch_size=8, shuffle=False)
result_file_types = ['txt']
result_filename = 'predictions'
break_step = 3


a,b = eval(model_id, test_loader, result_file_types, result_filename, break_step)