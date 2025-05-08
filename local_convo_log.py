import csv
from datetime import datetime
from mistral_inference.transformer import Transformer
from mistral_inference.generate import generate
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_common.protocol.instruct.messages import UserMessage
from mistral_common.protocol.instruct.request import ChatCompletionRequest

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

def save_to_csv(user_input, response):
    # Define the CSV file name
    csv_file = 'conversation_log.csv'

    # Get the current timestamp
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # Data to be saved
    data = [timestamp, user_input, response]

    # Write to the CSV file
    with open(csv_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(data)

def main():

    while True:
        try:
            user_input = input("Prompt: ")
            if user_input.strip() == "":
                print("=====================")
                continue
            response = generate_response(user_input)
            print(response)
            print("=====================")
            save_to_csv(user_input, response)
        except KeyboardInterrupt:
            print("\nExiting...")
            break

if __name__ == '__main__':
    main()
