import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import torch.utils.data

# Load tokenized data
input_ids, attention_masks, token_type_ids, labels = torch.load("tokenized_data.pt", weights_only=True)

# Define a PyTorch dataset class
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, input_ids, attention_masks, labels):
        self.input_ids = input_ids
        self.attention_masks = attention_masks
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_masks[idx],
            "labels": self.labels[idx],
        }

# Create the dataset
dataset = CustomDataset(input_ids, attention_masks, labels)

# Split dataset into train and validation sets (same as in fine-tuning)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

# Load the fine-tuned model and tokenizer
model = AutoModelForSequenceClassification.from_pretrained("./fine_tuned_deberta")
tokenizer = AutoTokenizer.from_pretrained("./fine_tuned_deberta")

# Define training arguments for evaluation
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",  # Evaluate at the end of each epoch
    per_device_eval_batch_size=8,
    logging_dir="./logs",
)

# Define metrics function
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = torch.argmax(torch.tensor(logits), dim=1)
    precision = precision_score(labels, predictions, average="binary")
    recall = recall_score(labels, predictions, average="binary")
    f1 = f1_score(labels, predictions, average="binary")
    accuracy = accuracy_score(labels, predictions)
    return {"precision": precision, "recall": recall, "f1": f1, "accuracy": accuracy}

# Initialize the Trainer for evaluation
trainer = Trainer(
    model=model,
    args=training_args,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)

# Perform evaluation
eval_results = trainer.evaluate()

# Print evaluation results
print("Evaluation Results:")
for key, value in eval_results.items():
    print(f"{key}: {value:.4f}")