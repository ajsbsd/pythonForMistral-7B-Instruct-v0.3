#####
#####
#####
from datasets import load_dataset

# Load the JSONL file
dataset = load_dataset('json', data_files='/notebooks/training_data.jsonl')

# If you want to specify a split (e.g., train, validation, test), you can do so like this:
#dataset = load_dataset('json', data_files={'train': 'path/to/your/training_data.jsonl'})
# Access the training split
train_dataset = dataset['train']

# Print the first few examples
for i in range(5):
    print(train_dataset[i])

    # Push the dataset to the Hugging Face Hub
dataset.push_to_hub('ajsbsd/live_speech')


