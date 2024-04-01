import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments

from get_logger import logger
from preprocessor import Preprocessor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Running with {device} device.")

tokenizer = RobertaTokenizer.from_pretrained('distilroberta-base')
model = RobertaForSequenceClassification.from_pretrained('distilroberta-base', num_labels=1).to(
    device)

original_data_path = "../data/Software.json"
reviews = pd.read_json(original_data_path, lines=True).sample(frac=0.25, random_state=1)
reviews = Preprocessor.clean_review_objects(reviews)
reviews = Preprocessor.standardize_columns(reviews, ["vote"])
dataset = reviews[['reviewText', 'voteStd']].rename(columns={'reviewText': 'text', 'voteStd': 'target'})
train_dataset, val_dataset = train_test_split(dataset, test_size=0.2, random_state=1)


def tokenize_and_encode(examples):
    return tokenizer(examples, padding="max_length", truncation=True, max_length=512)


train_encodings = tokenize_and_encode(train_dataset['text'].tolist())
val_encodings = tokenize_and_encode(val_dataset['text'].tolist())
train_encodings['labels'] = train_dataset['target'].tolist()
val_encodings['labels'] = val_dataset['target'].tolist()


class RegressionDataset(Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(item['labels'], dtype=torch.float)
        return item

    def __len__(self):
        return len(self.encodings['input_ids'])


training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=64,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    evaluation_strategy='epoch',
    fp16=True,
    gradient_accumulation_steps=3
)

train_dataset = RegressionDataset(train_encodings)
val_dataset = RegressionDataset(val_encodings)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

trainer.train()
model.save_pretrained('../models/fine_tuned_distilroberta')
