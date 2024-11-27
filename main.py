import pandas as pd
import torch
from transformers import AutoTokenizer

# Step 1: Load the dataset
file_path = "Cleaned_Optimized_Balanced_Training_Dataset.csv"  # Replace with your file path
dataset = pd.read_csv(file_path)

# Step 2: Load the tokenizer
model_name = "microsoft/deberta-v3-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Step 3: Tokenize the dataset
def tokenize_function(row):
    return tokenizer(
        row["unit_message"],        # First input
        row["DVIR"],                # Second input
        truncation=True,
        padding="max_length",
        max_length=128,
        return_tensors="pt"
    )

# Apply tokenization to the dataset
tokenized_dataset = dataset.apply(tokenize_function, axis=1)

# Step 4: Prepare tokenized data for PyTorch
input_ids = torch.cat([item['input_ids'] for item in tokenized_dataset.values])
attention_masks = torch.cat([item['attention_mask'] for item in tokenized_dataset.values])
token_type_ids = torch.cat([item['token_type_ids'] for item in tokenized_dataset.values])
labels = torch.tensor(dataset["duplicate"].values)

# Verify the shape of tokenized data
print(f"Input IDs shape: {input_ids.shape}")
print(f"Attention Masks shape: {attention_masks.shape}")
print(f"Labels shape: {labels.shape}")

# Step 5: Save tokenized data for training
torch.save((input_ids, attention_masks, token_type_ids, labels), "tokenized_data.pt")
print("Tokenized data saved as 'tokenized_data.pt'")