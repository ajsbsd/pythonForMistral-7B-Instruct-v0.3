from datasets import load_dataset

# Login using e.g. `huggingface-cli login` to access this dataset
ds = load_dataset("ajsbsd/gutenburg_little_cuban_rebel")
#pip install --upgrade typing_extensions
#pip install --upgrade transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
import csv
from datetime import datetime
from mistral_inference.transformer import Transformer
from mistral_inference.generate import generate
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_common.protocol.instruct.messages import UserMessage
from mistral_common.protocol.instruct.request import ChatCompletionRequest
import json
import os
#!huggingface-cli whoami


from datasets import load_dataset


# Load the JSONL file
#dataset = load_dataset('json', data_files='/notebooks/training_l')

# If you want to specify a split (e.g., train, validation, test), you can do so like this:
#dataset = load_dataset('json', data_files={'train': 'path/to/your/training_data.jsonl'})
#dataset = load_dataset('ajsbsd/live_speech')
from datasets import load_dataset

model_name = 'mistralai/Mistral-7B-Instruct-v0.3'  # Replace with the actual model name if different
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

def tokenize_function(examples):
    return tokenizer(examples['text'], padding="max_length", truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=4,
    save_steps=10_000,
    save_total_limit=2,
)

from transformers import Trainer

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
)

trainer.train()

trainer.save_model('./fine-tuned-model')
