import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments

from get_logger import logger
from preprocessor import Preprocessor

"""
Fine-tunes a DistilRoBERTa model for 'reviewText' -> 'voteStd' regression and saves it to the models directory.
"""

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Running with {device} device.")

tokenizer = RobertaTokenizer.from_pretrained("distilroberta-base")
model = RobertaForSequenceClassification.from_pretrained("distilroberta-base", num_labels=1).to(
    device)

data_path = "../data/Software.json"
reviews = pd.read_json(data_path, lines=True)
reviews, _ = train_test_split(reviews, test_size=0.25, random_state=1)
reviews = Preprocessor.clean_review_objects(reviews)
reviews = Preprocessor.standardize_columns(reviews, ["vote"])
input_data = reviews[["reviewText", "voteStd"]].rename(columns={"reviewText": "text", "voteStd": "target"})
training_data, validation_data = train_test_split(input_data, test_size=0.2, random_state=1)


def tokenize_and_encode(examples):
    return tokenizer(examples, padding="max_length", truncation=True, max_length=512)


training_encodings = tokenize_and_encode(training_data["text"].tolist())
training_encodings["labels"] = training_data["target"].tolist()

validation_encodings = tokenize_and_encode(validation_data["text"].tolist())
validation_encodings["labels"] = validation_data["target"].tolist()


class RegressionDataset(Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, index):
        item = {key: torch.tensor(value[index]) for key, value in self.encodings.items()}
        item["labels"] = torch.tensor(item["labels"], dtype=torch.float)
        return item

    def __len__(self):
        return len(self.encodings["input_ids"])


training_args = TrainingArguments(
    output_dir="../results",
    num_train_epochs=3,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=64,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="../logs",
    logging_steps=10,
    evaluation_strategy="epoch",
    fp16=True,
    gradient_accumulation_steps=3
)

training_data = RegressionDataset(training_encodings)
validation_data = RegressionDataset(validation_encodings)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=training_data,
    eval_dataset=validation_data,
)

trainer.train()
model.save_pretrained("../models/fine_tuned_distilroberta")
