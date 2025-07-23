# fine_tune_t5.py

import pandas as pd
from datasets import Dataset
from transformers import T5Tokenizer, T5ForConditionalGeneration, TrainingArguments, Trainer

# Load dataset
df = pd.read_csv("idioms_dataset.csv")
df["input_text"] = "translate Hindi to English: " + df["source"]
df["target_text"] = df["target"]
dataset = Dataset.from_pandas(df[["input_text", "target_text"]])

# Tokenizer and model
model_name = "t5-small"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

def preprocess(examples):
    inputs = tokenizer(examples["input_text"], padding="max_length", truncation=True, max_length=64)
    targets = tokenizer(examples["target_text"], padding="max_length", truncation=True, max_length=64)
    inputs["labels"] = targets["input_ids"]
    return inputs

tokenized_data = dataset.map(preprocess, batched=True)

# Training Arguments
training_args = TrainingArguments(
    output_dir="./t5_cultural_idioms",
    per_device_train_batch_size=2,
    num_train_epochs=10,
    save_steps=10,
    logging_dir="./logs",
    logging_steps=5
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_data,
    tokenizer=tokenizer
)

# Train model
trainer.train()

# Save model
trainer.save_model("./t5_cultural_idioms")
tokenizer.save_pretrained("./t5_cultural_idioms")
