import os
import torch
from datasets import load_dataset
from transformers import (
    DistilBertForTokenClassification,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)

from src import data, metrics, tokens

# Load the FiNER-139 dataset
dataset = load_dataset("nlpaueb/finer-139")

# Preprocessing the data
dataset = data.process(dataset)  # apply our filters
tokenized_datasets = dataset.map(tokens.tokenize_and_align_labels, batched=True, num_proc=os.cpu_count())

# Model configuration
num_labels = dataset["train"].features["ner_tags"].feature.num_classes
id2label = {i: label for i, label in enumerate(["O"] + data.TAG_NAMES)}
label2id = {label: i for i, label in enumerate(["O"] + data.TAG_NAMES)}

print(f"Training for: {id2label}")

model = DistilBertForTokenClassification.from_pretrained(
    "distilbert-base-cased",
    num_labels=num_labels,
    id2label=id2label,
    label2id=label2id,
)

data_collator = DataCollatorForTokenClassification(tokenizer=tokens.tokenizer)

# Training the model
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=8,
    num_train_epochs=20,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="eval_f1",
    greater_is_better=True,
)


class ClassWeightTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        freq = [1356091, 11445, 15208, 11693, 2338]  # class frequencies from EDA
        total = sum(freq)
        self.__class_weights = [total / (len(freq) * f + 1e-6) for f in freq]
        super().__init__(*args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        # forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")
        # compute custom loss (suppose one has 2 labels with different weights)
        loss_fct = torch.nn.CrossEntropyLoss(weight=torch.tensor(self.__class_weights, device=model.device))
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    compute_metrics=metrics.compute_metrics,
    tokenizer=tokens.tokenizer,
    callbacks=[
        EarlyStoppingCallback(early_stopping_patience=2),
    ],
)

trainer.train()

# Evaluate the model on the test set
eval_results = trainer.predict(tokenized_datasets["test"])
print("Evaluation metrics on the test set:")
print(eval_results.metrics)

trainer.save_model("./model/finer-debt-distilbert-cased")
trainer.save_metrics("test", eval_results.metrics)

metrics.save_confusion_matrix(eval_results)
print(metrics.report(eval_results))
