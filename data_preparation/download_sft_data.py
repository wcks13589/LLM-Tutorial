import os
import json
import random
from datasets import load_dataset

dataset = load_dataset('erhwenkuo/alpaca-data-gpt4-chinese-zhtw')['train']
output_path = 'data/alpaca/gpt4-chinese-zhtw.jsonl'
os.makedirs(os.path.dirname(output_path), exist_ok=True)

with open(output_path, 'w') as f:
    for human_instruction, human_input, assistant_output in zip(dataset['instruction'], dataset['input'], dataset['output']):
        f.write(json.dumps({'input': '\n'.join([human_instruction.strip(),human_input.strip()]).strip(), 'output': assistant_output.strip()}, ensure_ascii=False)+ '\n')

input_file = "data/alpaca/gpt4-chinese-zhtw.jsonl"
training_output_file = "data/alpaca/training.jsonl"
validation_output_file = "data/alpaca/validation.jsonl"
test_output_file = "data/alpaca/test.jsonl"

# Specify the proportion of data for training and validation
train_proportion = 0.98
validation_proportion = 0.01
test_proportion = 0.01

# Read the JSONL file and shuffle the JSON objects
with open(input_file, "r") as f:
    lines = f.readlines()
    random.shuffle(lines)

# Calculate split indices
total_lines = len(lines)
train_index = int(total_lines * train_proportion)
val_index = int(total_lines * validation_proportion)

# Distribute JSON objects into training and validation sets
train_data = lines[:train_index]
validation_data = lines[train_index:train_index+val_index]
test_data = lines[train_index+val_index:]

# Write JSON objects to training file
with open(training_output_file, "w") as f:
    for line in train_data:
        f.write(line.strip() + "\n")

# Write JSON objects to validation file
with open(validation_output_file, "w") as f:
    for line in validation_data:
        f.write(line.strip() + "\n")

# Write JSON objects to training file
with open(test_output_file, "w") as f:
    for line in test_data:
        f.write(line.strip() + "\n")