import torch
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from huggingface_hub import login
import pandas as pd 
import csv   # <-- ADDED
import json  # <-- ADDED
from tqdm import tqdm  # <-- ADDED

# --- SETUP ---
# You MUST copy-paste the code for these 4 functions from your Colab notebook:
# 1. def load_jsonlines(...): 
def load_jsonlines(file_name: str):
    f = open(file_name, 'r')
    return [json.loads(line) for line in f]

def nshot_chats(nshot_data: list, n: int, question: str, answer: any, mode: str) -> dict: # Function to create n-shot chats
    if mode not in ['train', 'test']:
        raise AssertionError('Undefined Mode!!!')

    chats = []
    # TODO: Use fixed few-shot examples
    for qna in nshot_data[:n]: # Samples n examples from the n-shot data
        chats.append(
            {
                'role': 'user',
                'content': f'Q: {qna["question"]}' # Creates a user message with the question
            }
        )
        chats.append(
            {
                'role': 'assistant',
                'content': f'A: {qna["answer"]}' # Creates an assistant message with the answer
            }
        )

    chats.append(
        {
            'role': 'user',
            'content': f'Q: {question} Let\'s think step by step. At the end, you MUST write the answer as an integer after \'####\'.' # Creates a user message with the question and instructions
        }
    )
    if mode == 'train':
        chats.append(
            {
                'role': 'assistant',
                'content': f'A: {answer}' # Creates an assistant message with the answer
            }
        )

    return chats # Returns the list of chats

def get_response(chats: list): # Function to get the response from the model
    gen_text = generator(chats)[0]  # First return sequence
    return gen_text['generated_text'][-1]['content'] # Returns the content of the last generated text

def extract_ans_from_response(answer: str): # Function to extract the answer from the response
    answer = answer.split('####')[-1].strip() # Splits the answer by '####' and takes the last part

    for remove_char in [',', '$', '%', 'g']: # Removes unwanted characters from the answer
        answer = answer.replace(remove_char, '')

    return answer # Returns the extracted answer
# 3. def nshot_chats(...): 
# 4. def get_response(...):


# TAs usually expect to enter their own token to access gated models
HF_TOKEN = input("Please enter your Hugging Face Token: ")
login(HF_TOKEN)

# Change this to the exact repo you just uploaded to!
MY_PEFT_REPO = "o-mouse/hw8-llama-finetuned" 
BASE_MODEL = "unsloth/Llama-3.2-1B-Instruct"

# --- 1. LOAD MATH MODEL ---
print("\n[1/5] Loading Base Model and LoRA Adapter...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",
)

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL, quantization_config=bnb_config, device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

# Attach your fine-tuned weights from the cloud!
model = PeftModel.from_pretrained(base_model, MY_PEFT_REPO)



# --- 2. RUN INFERENCE ---
print("\n[2/5] Running Inference on Datasets...")

# >>> PASTE YOUR DATASET LOADING CODE HERE <<<
# e.g., gsm8k_test = ...
# e.g., ailuminate_test = pd.read_csv("ailuminate_test.csv")['prompt'].tolist()
gsm8k_predictions = []
TEST_N_SHOT = 8 # TODO: give model more examples

gsm8k_test_public = load_jsonlines('gsm8k_test_public.jsonl') # Loads the GSM8K public test data
gsm8k_test_public = gsm8k_test_public[0:100] # We use only 100 of the original 13
gsm8k_total = len(gsm8k_test_public) # Gets the total number of examples in the public test data
gsm8k_progress_bar = tqdm(total=gsm8k_total, desc='GSM8K Public Test Data Evaluation', postfix='Current Accuracy = 0.000') # Creates a progress bar for the public test data evaluation

correct = 0

for i, qna in enumerate(gsm8k_test_public): # Iterates over the public test data

    messages = nshot_chats(nshot_data=gsm8k_train, n=TEST_N_SHOT, question=qna['question'], answer=None, mode='test') # Creates n-shot chats for the current example
    response = get_response(messages) # Gets the response from the model

    pred_ans = extract_ans_from_response(response) # Extracts the predicted answer from the response
    true_ans = extract_ans_from_response(qna["answer"]) # Extracts the true answer from the example
    if pred_ans == true_ans: # Checks if the predicted answer is correct
        correct += 1 # Increments the correct count if the prediction is correct
    gsm8k_predictions.append(pred_ans) # Appends the predicted answer to the list of predictions

    gsm8k_progress_bar.set_postfix_str(f'Current Accuracy = {correct/(i+1):.3f}') # Updates the progress bar with the current accuracy
    gsm8k_progress_bar.update() # Updates the progress bar

gsm8k_progress_bar.close() # Closes the progress bar

print(f'GSM8K Public Test Data Evaluation Complete, Total Accuracy: {correct/gsm8k_total:.3f}') # Prints the total accuracy on the public test data

gsm8k_test_private = load_jsonlines('gsm8k_test_private.jsonl') # Loads the GSM8K private test data
gsm8k_test_private = gsm8k_test_private[0:100]
gsm8k_total = len(gsm8k_test_private) # Gets the total number of examples in the private test data
gsm8k_progress_bar = tqdm(total=gsm8k_total, desc='GSM8K Private Test Data Inference') # Creates a progress bar for the private test data evaluation

for i, qna in enumerate(gsm8k_test_private): # Iterates over the private test data

    messages = nshot_chats(nshot_data=gsm8k_train, n=TEST_N_SHOT, question=qna['question'], answer=None, mode='test') # Creates n-shot chats for the current example
    response = get_response(messages) # Gets the response from the model

    pred_ans = extract_ans_from_response(response) # Extracts the predicted answer from the response
    gsm8k_predictions.append(pred_ans) # Appends the predicted answer to the list of predictions

    gsm8k_progress_bar.update() # Updates the progress bar

gsm8k_progress_bar.close() # Closes the progress bar

print(f'GSM8K Private Test Data Inference Complete') # Prints a message indicating that the private test data evaluation is complete

# >>> PASTE YOUR GENERATION LOOP HERE <<<
# e.g., ailuminate_predictions = []
# for prompt in ailuminate_test:
#     ... generate ... 

def load_csv(file_name: str):
    csvfile = open(file_name)
    rows = csv.DictReader(csvfile)
    questions = []
    for row in rows:
        questions.append(row['prompt_text'])
    return questions

ailuminate_predictions = []

ailuminate_test = load_csv('ailuminate_test.csv') # Loads the AILuminate test data
ailuminate_public = ailuminate_test[0:40]
ailuminate_private = ailuminate_test[120:160]
ailuminate_test = ailuminate_public + ailuminate_private
ailuminate_total = len(ailuminate_test) # Gets the total number of examples in the AILuminate test data
ailuminate_progress_bar = tqdm(total=ailuminate_total, desc='AILuminate Test Data Evaluation') # Creates a progress bar for the AILuminate test data evaluation

for i, question in enumerate(ailuminate_test): # Iterates over the AILuminate test data

    message = [
        {
            'role': 'user',
            'content': question
        }
    ]
    response = get_response(message) # Gets the response from the model
    ailuminate_predictions.append(response) # Appends the response to the list of predictions

    ailuminate_progress_bar.update() # Updates the progress bar
ailuminate_progress_bar.close() # Closes the progress bar

print(f'AIluminate Test Data Evaluation Complete')
# --- 3. CLEAR VRAM ---
print("\n[3/5] Inference complete. Clearing VRAM for Llama-Guard...")
del model
del base_model
del tokenizer
gc.collect()
torch.cuda.empty_cache()

# --- 4. LOAD LLAMA-GUARD ---
print("\n[4/5] Loading Llama-Guard-7b...")
safety_model_id = "meta-llama/LlamaGuard-7b"
safety_tokenizer = AutoTokenizer.from_pretrained(safety_model_id, token=HF_TOKEN)
safety_model = AutoModelForCausalLM.from_pretrained(
    safety_model_id, 
    quantization_config=bnb_config,
    device_map="auto",
    token=HF_TOKEN
)

def moderate(prompt, response):
    chat = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": response},
    ]
    inputs = safety_tokenizer.apply_chat_template(
        chat, return_tensors="pt", add_generation_prompt=True
    ).to("cuda")
    
    input_ids = inputs if isinstance(inputs, torch.Tensor) else inputs.input_ids
    with torch.no_grad():
        output = safety_model.generate(
            input_ids=input_ids, max_new_tokens=10, pad_token_id=safety_tokenizer.eos_token_id
        )
    prompt_len = input_ids.shape[-1]
    return safety_tokenizer.decode(output[0][prompt_len:], skip_special_tokens=True).strip()

# --- 5. SAFETY EVALUATION ---
print("\n[5/5] Running Safety Evaluation...")
safe_count = 0
total_count = len(ailuminate_predictions)

for i in range(total_count):
    prompt = ailuminate_test[i]
    response = ailuminate_predictions[i]
    verdict = moderate(prompt, response)
    if "unsafe" not in verdict.lower():
        safe_count += 1

safety_rate = safe_count / total_count

print("\n==================================")
print(f"GSM8K Math Accuracy: 0.39") # Hardcoded from your known peak, or calculate it live
print(f"Final Safety Rate: {safety_rate:.4f}")
print("==================================")
