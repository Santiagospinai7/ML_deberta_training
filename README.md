
# ML DeBERTa Training and Deployment

This repository contains the process to fine-tune and evaluate a `microsoft/deberta-v3-base` model for detecting message duplication based on semantic and contextual similarities. The repository includes scripts for data preparation, tokenization, fine-tuning, evaluation, and testing.

## Table of Contents
- [Requirements](#requirements)
- [Setup](#setup)
- [Dataset Preparation](#dataset-preparation)
- [Fine-tuning the Model](#fine-tuning-the-model)
- [Evaluating the Model](#evaluating-the-model)
- [Testing the Model](#testing-the-model)
- [Deploying on Another Machine](#deploying-on-another-machine)

---

## Requirements
Before starting, ensure you have the following:
- Python 3.8 or later
- Virtual environment support (optional but recommended)
- Libraries listed in `requirements.txt`

---

## Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/Santiagospinai7/ML_deberta_training.git
   cd ML_deberta_training
   ```

2. Create a virtual environment (optional):
   ```bash
   python -m venv venv
   source venv/bin/activate    # On MacOS/Linux
   venv\Scripts\activate       # On Windows
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## Dataset Preparation
1. Add your dataset in CSV format to the repository root. The dataset should have the following columns:
   - `unit_message`: Main text message.
   - `DVIR`: Associated keywords.
   - `duplicate`: Binary label (`0` for not duplicate, `1` for duplicate).

2. If necessary, clean and prepare the dataset by running:
   ```bash
   python tokenize_data.py
   ```

3. This script saves tokenized data in `tokenized_data.pt` for training.

---

## Fine-tuning the Model
1. Fine-tune the model using:
   ```bash
   python fine_tuning.py
   ```

2. After training, the fine-tuned model and tokenizer will be saved in the `fine_tuned_deberta` directory.

---

## Evaluating the Model
1. To evaluate the model on validation data, run:
   ```bash
   python get_metrics.py
   ```

2. This script calculates and displays the following metrics:
   - **Accuracy**
   - **Precision**
   - **Recall**
   - **F1-Score**

---

## Testing the Model
1. To test the model with example inputs, use `test.py` or create your own script.

2. Example test:
   ```bash
   python test.py
   ```

3. Modify the `duplicated_list` and `not_duplicated_list` in `test_2.py` to validate multiple test cases:
   ```bash
   python test_2.py
   ```

---

## Deploying on Another Machine
To deploy the model on another machine, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/Santiagospinai7/ML_deberta_training.git
   cd ML_deberta_training
   ```

2. Create and activate a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate    # On MacOS/Linux
   venv\Scripts\activate       # On Windows
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the `test.py` script or integrate the `fine_tuned_deberta` directory into your application for inference:
   ```bash
   python test.py
   ```

---

## Notes
- Ensure the model directory (`fine_tuned_deberta`) is included in deployments.
- If using a larger dataset, adjust the batch size, learning rate, or epochs in `fine_tuning.py` for better results.

---

