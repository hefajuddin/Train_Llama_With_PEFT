from transformers import TrainingArguments
from transformers import Trainer
from src.pretrained_model import pretrained_model
from src.tokenized_model import tokenized_dataset
from src.collator import data_collator
from src.tokenized_model import tokenizer

from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir="./llama2-finetuned",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    weight_decay=0.01,
    save_steps=10_000,
    save_total_limit=2,
    fp16=True,
    push_to_hub=False,
    logging_dir="./logs"
)

trainer = Trainer(
    model=pretrained_model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
)