from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

duplicated_list = [
  ["LR TIRES WORN / DONE MASGWI 6/27 MTB", 'flat tires'],
  ["Driver ID: 'FOUED', Defect Type: 'FRAME/BODY', Description: 'DAMAGED', Remarks: '1. HOLE PIC TAKEN AND SENT TO BETH P.'", 'damaged frame/body'],
  ["Driver ID: 'KLEMI', Defect Type: 'TIRES', Description: 'DAMAGED', Remarks: '1. IRREGULAR WEAR PATTERN'", 'tread depth tires'],
  ["Lubed up tandems and verified pins are all releasing. Performed dot. DOT failed. LFI and LFO tires low on tread and cupping on inside edges. RRO tire has flat spot. RFI tire cupping on inside edge. RRI tire flat. All 4 torque arm bushings bad and should be replaced, causing uneven tire wear. Mud flap bracket mounts are cracked. All paint peeled off license plate, cannot read plate. Left side crossmember bent and cracked above tandems. All 4 s cam bushings have excessive play, fronts worse than rear. RR brake chamber damaged, plastic guide sticking out from push rod hole. Left dolly leg lower support band broken. Bill box cover missing. We replaced RRI AND RRO tires so unit could safely be brought back to shop. Unit needs about 5 tires. Attached are photos. Please let me know if you would like us to pick up this trailer?", 'damaged frame/body']
  ["DRIVER SIDE MUDFLAP, HANGERS BENT", "bracket damaged mud flaps"]
]

not_duplicated_list = [
  ["LR TIRES WORN / DONE MASGWI 6/27 MTB", 'flat tires'],
  ["Driver ID: 'FOUED', Defect Type: 'FRAME/BODY', Description: 'DAMAGED', Remarks: '1. HOLE PIC TAKEN AND SENT TO BETH P.'", 'damaged frame/body'],
  ["Driver ID: 'KLEMI', Defect Type: 'TIRES', Description: 'DAMAGED', Remarks: '1. IRREGULAR WEAR PATTERN'", 'tread depth tires'],
]

# Load the fine-tuned model and tokenizer
model = AutoModelForSequenceClassification.from_pretrained("./fine_tuned_deberta")
tokenizer = AutoTokenizer.from_pretrained("./fine_tuned_deberta")

# Test example
unit_message = ""
dvir = ""

# Tokenize input
inputs = tokenizer(
    unit_message, dvir, truncation=True, padding="max_length", max_length=128, return_tensors="pt"
)

# Perform inference
outputs = model(**inputs)
logits = outputs.logits
predicted_class = torch.argmax(logits, dim=1).item()  # 0 = Not duplicate, 1 = Duplicate

print(f"Prediction: {'Duplicate' if predicted_class == 1 else 'Not Duplicate'}")