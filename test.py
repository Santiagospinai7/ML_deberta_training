import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

# Load the cleaned dataset
csv_path = "Cleaned_Optimized_Balanced_Training_Dataset.csv"
data = pd.read_csv(csv_path)

# Extract examples
duplicated_examples = data[data["duplicate"] == 1].sample(5)  # 5 examples with duplicate = 1
not_duplicated_examples = data[data["duplicate"] == 0].sample(5)  # 5 examples with duplicate = 0

# Build duplicated and not duplicated lists
duplicated_list = [
    (row["unit_message"], row["DVIR"]) for _, row in duplicated_examples.iterrows()
]

not_duplicated_list = [
    (row["unit_message"], row["DVIR"]) for _, row in not_duplicated_examples.iterrows()
]

# Load the fine-tuned model and tokenizer
model = AutoModelForSequenceClassification.from_pretrained("./fine_tuned_deberta")
tokenizer = AutoTokenizer.from_pretrained("./fine_tuned_deberta")

# Define prediction function
def predict(unit_message, dvir):
    inputs = tokenizer(
        unit_message, dvir, truncation=True, padding="max_length", max_length=128, return_tensors="pt"
    )
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=1).item()  # 0 = Not duplicate, 1 = Duplicate
    return predicted_class

# Test the lists
def test_model(test_list, expected_label):
    predictions = []
    for unit_message, dvir in test_list:
        predicted_class = predict(unit_message, dvir)
        predictions.append(predicted_class)
        print(f"Message: {unit_message}\nDVIR: {dvir}\nPrediction: {'Duplicate' if predicted_class == 1 else 'Not Duplicate'}\n")
    
    correct = sum(1 for p in predictions if p == expected_label)
    print(f"Accuracy for label {expected_label}: {correct}/{len(test_list)} ({correct / len(test_list) * 100:.2f}%)")
    return predictions

# Run the tests
print("Testing duplicated list (Expected: Duplicate)")
duplicated_predictions = test_model(duplicated_list, expected_label=1)

print("\nTesting not duplicated list (Expected: Not Duplicate)")
not_duplicated_predictions = test_model(not_duplicated_list, expected_label=0)

# Validate lengths
print("\nValidation:")
print(f"Length of duplicated list: {len(duplicated_list)}")
print(f"Length of duplicated predictions: {len(duplicated_predictions)}")
print(f"Length of not duplicated list: {len(not_duplicated_list)}")
print(f"Length of not duplicated predictions: {len(not_duplicated_predictions)}")