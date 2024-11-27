import pandas as pd
from transformers import AutoTokenizer

# Step 1: Load the dataset
file_path = "Cleaned_Optimized_Balanced_Training_Dataset.csv"  # Replace with your actual file path
dataset = pd.read_csv(file_path)

# Step 2: Load the DeBERTa tokenizer
model_name = "microsoft/deberta-v3-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Step 3: Define the tokenization function
def tokenize_function(row):
    return tokenizer(
        row["unit_message"],        # First input: the main message
        row["DVIR"],                # Second input: the keywords
        truncation=True,            # Truncate inputs longer than max_length
        padding="max_length",       # Pad inputs to max_length
        max_length=128,             # Maximum sequence length
        return_tensors="pt"         # Return PyTorch tensors
    )

# Step 4: Apply tokenization to the dataset
dataset_tokenized = dataset.apply(tokenize_function, axis=1)

# Step 5: Inspect the first few tokenized examples
print(dataset_tokenized.head())