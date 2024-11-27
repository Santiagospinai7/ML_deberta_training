#	1.	Dataset Preparation:
#   Loads the tokenized data and creates a PyTorch dataset.
#   Splits the data into 80% training and 20% validation.
# 2.	Model Initialization:
# 	Loads microsoft/deberta-v3-base and adapts it for binary classification (labels: 0 and 1).
# 3.	Training:
# 	Uses Hugging Face’s Trainer to fine-tune the model.
# 	Monitors evaluation metrics (precision, recall, F1-score, accuracy).
# 4.	Save the Model:
#	  Saves the fine-tuned model and tokenizer for later use.

import torch
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments, AutoTokenizer

# Load tokenized data
# input_ids, attention_masks, token_type_ids, labels = torch.load("tokenized_data.pt")
input_ids, attention_masks, token_type_ids, labels = torch.load("tokenized_data.pt", weights_only=True)

# Convert data to PyTorch Dataset
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

dataset = CustomDataset(input_ids, attention_masks, labels)

# Split into training and validation sets
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

# Load the model
model = AutoModelForSequenceClassification.from_pretrained("microsoft/deberta-v3-base", num_labels=2)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",  # Evaluate at the end of each epoch
    save_strategy="epoch",        # Save at the end of each epoch
    learning_rate=5e-5,
    per_device_train_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    save_total_limit=2,
    logging_dir="./logs",
    logging_steps=10,
    load_best_model_at_end=True,  # Load the best model based on evaluation metrics
    metric_for_best_model="accuracy",
)

# Define metrics for evaluation
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = torch.argmax(torch.tensor(logits), dim=1)
    precision = precision_score(labels, predictions, average="binary")
    recall = recall_score(labels, predictions, average="binary")
    f1 = f1_score(labels, predictions, average="binary")
    accuracy = accuracy_score(labels, predictions)
    return {"precision": precision, "recall": recall, "f1": f1, "accuracy": accuracy}

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)

# Fine-tune the model
trainer.train()

# Save the fine-tuned model
model.save_pretrained("./fine_tuned_deberta")
tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")
tokenizer.save_pretrained("./fine_tuned_deberta")
print("Fine-tuning complete. Model and tokenizer saved!")