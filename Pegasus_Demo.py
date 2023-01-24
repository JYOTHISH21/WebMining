
import pandas as pd
from datasets import load_dataset
dataset = load_dataset("xsum")
in_df = pd.read_csv(dataset)

# Train Test Split
train_pct = 0.6
test_pct = 0.2

in_df = in_df.sample(len(in_df), random_state=20)
train_sub = int(len(in_df) * train_pct)
test_sub = int(len(in_df) * test_pct) + train_sub

train_df = in_df[0:train_sub]
test_df = in_df[train_sub:test_sub]
val_df = in_df[test_sub:]

train_texts = list(train_df['allTextReprocess'])
test_texts = list(test_df['allTextReprocess'])
val_texts = list(val_df['allTextReprocess'])

train_decode = list(train_df['summaries'])
test_decode = list(test_df['summaries'])
val_decode = list(val_df['summaries'])

import transformers

import torch
min_length = 15
max_length = 40

# Setup model
model_name = 'google/pegasus-xsum'
torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
tokenizer = transformers.PegasusTokenizer.from_pretrained(model_name)

model = transformers.PegasusForConditionalGeneration.from_pretrained(model_name).to(torch_device)
in_text = [in_df['allTextReprocess'].iloc[3]]
batch = tokenizer.prepare_seq2seq_batch(in_text, truncation=True, padding='longest').to(torch_device) 

translated = model.generate(min_length=min_length, max_length=max_length, **batch)
tgt_text0 = tokenizer.batch_decode(translated, skip_special_tokens=True)
print(tgt_text0)

# Tokenize
train_encodings = tokenizer(train_texts, truncation=True, padding=True)
val_encodings = tokenizer(val_texts, truncation=True, padding=True)
test_encodings = tokenizer(test_texts, truncation=True, padding=True)

train_labels = tokenizer(train_decode, truncation=True, padding=True)
val_labels = tokenizer(val_decode, truncation=True, padding=True)
test_labels = tokenizer(test_decode, truncation=True, padding=True)

# Setup dataset objects
class Summary_dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels['input_ids'][idx])  # torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.encodings)

train_dataset = Summary_dataset(train_encodings, train_labels)
val_dataset = Summary_dataset(val_encodings, val_labels)
test_dataset = Summary_dataset(test_encodings, test_labels)

# Training
from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir='./results',          # output directory
    num_train_epochs=1000,              # total number of training epochs
    per_device_train_batch_size=16,  # batch size per device during training
    per_device_eval_batch_size=64,   # batch size for evaluation
    warmup_steps=500,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir='./logs',            # directory for storing logs
    logging_steps=10,
)

trainer = Trainer(
    model=model,                         # the instantiated 🤗 Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=train_dataset,         # training dataset
    eval_dataset=val_dataset             # evaluation dataset
)

trainer.train()

# Check results
in_text = [in_df['allTextReprocess'].iloc[3]]
batch = tokenizer.prepare_seq2seq_batch(in_text, truncation=True, padding='longest').to(torch_device) 

translated = model.generate(min_length=min_length, max_length=max_length, **batch)
tgt_text = tokenizer.batch_decode(translated, skip_special_tokens=True)
print(tgt_text)