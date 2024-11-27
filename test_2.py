from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

# Load the fine-tuned model and tokenizer
model = AutoModelForSequenceClassification.from_pretrained("./fine_tuned_deberta")
tokenizer = AutoTokenizer.from_pretrained("./fine_tuned_deberta")

# Test dataset
test_data = [
    # Format: [unit_message, dvir, true_label]
    ["The left tire is flat and needs replacement", "flat tires", 1],  # True positive
    ["The trailer lights are functional", "flat tires", 0],  # True negative
    ["Brake light is not working", "damaged frame/body", 0],  # False positive
    ["Brake chamber is leaking", "brake issue", 1],  # True positive
]

# Initialize lists to store results
true_labels = []
predicted_labels = []

# Perform inference for each example in the test dataset
for unit_message, dvir, true_label in test_data:
    inputs = tokenizer(
        unit_message, dvir, truncation=True, padding="max_length", max_length=128, return_tensors="pt"
    )
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=1).item()  # 0 = Not duplicate, 1 = Duplicate
    
    true_labels.append(true_label)
    predicted_labels.append(predicted_class)

# Calculate metrics
accuracy = accuracy_score(true_labels, predicted_labels)
precision = precision_score(true_labels, predicted_labels, average="binary")
recall = recall_score(true_labels, predicted_labels, average="binary")
f1 = f1_score(true_labels, predicted_labels, average="binary")

# Display results
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")