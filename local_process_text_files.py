#####
#####
##### Process Text Files
#####
#####
pip install --upgrade typing_extensions
import csv
from datetime import datetime
from mistral_inference.transformer import Transformer
from mistral_inference.generate import generate
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_common.protocol.instruct.messages import UserMessage
from mistral_common.protocol.instruct.request import ChatCompletionRequest
import json
import os

# Initialize Mistral model and tokenizer
mistral_models_path = "/root/mistral_models/7B-Instruct-v0.3"  # Update this path as necessary
tokenizer = MistralTokenizer.from_file(f"{mistral_models_path}/tokenizer.model.v3")
model = Transformer.from_folder(mistral_models_path)

def generate_response(user_input):
    completion_request = ChatCompletionRequest(messages=[UserMessage(content=user_input)])
    tokens = tokenizer.encode_chat_completion(completion_request).tokens
    out_tokens, _ = generate([tokens], model, max_tokens=64, temperature=0.0, eos_id=tokenizer.instruct_tokenizer.tokenizer.eos_id)
    result = tokenizer.instruct_tokenizer.tokenizer.decode(out_tokens[0])
    return result


def chunk_text_file(filepath, chunk_size=1000):
    with open(filepath, 'r', encoding='utf-8') as f:
        text = f.read()

    chunks = []
    i = 0
    while i < len(text):
        chunk = text[i:i + chunk_size]
        chunks.append(chunk.strip())
        i += chunk_size

    return chunks

def make_prompt_response_pairs(chunks):
    pairs = []
    for i, chunk in enumerate(chunks):
        prompt = f"Please explain the following in simple terms:\n\n{chunk}"
        print(f"[{i+1}/{len(chunks)}] Generating response...")
        response = generate_response(prompt)
        pairs.append({
            "prompt": prompt,
            "response": response.strip()
        })
    return pairs

def save_as_jsonl(pairs, output_file='training_data.jsonl'):
    with open(output_file, 'w', encoding='utf-8') as f:
        for pair in pairs:
            json.dump(pair, f, ensure_ascii=False)
            f.write('\n')

def main():
    input_file = input("Enter path to textbook .txt file: ").strip()
    if not os.path.isfile(input_file):
        print("File not found.")
        return

    chunk_size = 1000
    chunks = chunk_text_file(input_file, chunk_size)
    pairs = make_prompt_response_pairs(chunks)
    save_as_jsonl(pairs)
    print(f"\n✅ Done. Saved {len(pairs)} prompt/response pairs to training_data.jsonl")

if __name__ == '__main__':
    main()

