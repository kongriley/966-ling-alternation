# %%

from tqdm import tqdm
import os

import json

from dotenv import load_dotenv
load_dotenv('.env')

os.environ['HF_TOKEN'] = os.getenv('HF_TOKEN')

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

# %%

print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained("NousResearch/Llama-3.2-1B")
model = AutoModelForCausalLM.from_pretrained("NousResearch/Llama-3.2-1B", device_map="auto")
print("Model loaded")

# %%

def compute_average_logprobs(prefix, completion): 
    input_ids = tokenizer(prefix + " " + completion, return_tensors="pt").input_ids.to(model.device)

    with torch.no_grad():
        outputs = model(input_ids)
        log_probs = F.log_softmax(outputs.logits[0], dim=-1)
        total_logprobs = log_probs.gather(dim=-1, index=input_ids[0].unsqueeze(-1)).squeeze(-1).sum()

    # Normalize by the number of tokens
    average_logprobs = total_logprobs / input_ids.size(1)

    return average_logprobs.item()

def alternate_completions(prefix, completions):
    return [compute_average_logprobs(prefix, completion) for completion in completions]


with open('trials.json', 'r') as f:
    trials = json.load(f)

results = []
for trial in tqdm(trials):
    logprobs = alternate_completions(trial['prefix'], trial['completions'])
    prob = logprobs[1] / (logprobs[0] + logprobs[1])
    ratio = logprobs[1] / logprobs[0]

    results.append({
        'prefix': trial['prefix'],
        'logprobs': logprobs,
        'prob': prob,
        'ratio': ratio,
    })


with open('results.json', 'w') as f:
    json.dump(results, f, indent=4)

for result, trial in zip(results, trials):
    a, b = result['logprobs']
    if a < b:
        print(result['prefix'], trial['completions'][0])
    else:
        print(result['prefix'], trial['completions'][1])

# %%
