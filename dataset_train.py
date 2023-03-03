import datasets as ds
import torch
import numpy as np
from transformers import SwinForImageClassification, Trainer, TrainingArguments, AutoFeatureExtractor

dataset = ds.load_dataset("imagefolder", data_dir="./datasmall",cache_dir='./cache')
#print(dataset)
'''
#making sure it is working
ex=dataset['train'][400]
image=ex['image']

image.show()
labels = dataset['train'].features['label']
print(labels.int2str(ex['label']))
'''
access_token='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'
model_name= 'microsoft/swin-base-patch4-window12-384'
feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
def transform(example_batch):
    # Take a list of PIL images and turn them to pixel values
    inputs = feature_extractor([x.convert('RGB') for x in example_batch['image']], return_tensors='pt')
    inputs['label'] = example_batch['label']
    return inputs
prepared_ds = dataset.with_transform(transform)
def collate_fn(batch):
  #data collator
    return {
        'pixel_values': torch.stack([x['pixel_values'] for x in batch]),
        'labels': torch.tensor([x['label'] for x in batch])
    }
metric = ds.load_metric("accuracy")
def compute_metrics(p):
  # function which calculates accuracy for a certain set of predictions
  return metric.compute(predictions=np.argmax(p.predictions, axis=1), references=p.label_ids)

labels = dataset['train'].features['label'].names

# initialzing the model
model = SwinForImageClassification.from_pretrained(
    model_name,
    num_labels=len(labels),
    id2label={str(i): c for i, c in enumerate(labels)},
    label2id={c: str(i) for i, c in enumerate(labels)},
    ignore_mismatched_sizes = True,
)
model = model.to("cuda")
batch_size = 1
training_args = TrainingArguments(
    F'katuzar/swin-base-patch4-window12-384_77',
    remove_unused_columns=False,
    evaluation_strategy = "epoch",
    save_strategy = "epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=batch_size,
    gradient_accumulation_steps=4,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=5,
    warmup_ratio=0.1,
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    push_to_hub=False,
)

# Instantiate the Trainer object
trainer = Trainer(
    model=model,
    
    args=training_args,
    data_collator=collate_fn,
    compute_metrics=compute_metrics,
    train_dataset=prepared_ds["train"],
    eval_dataset=prepared_ds["validation"],
    tokenizer=feature_extractor,
)

train_results = trainer.train()
trainer.save_model()
trainer.log_metrics("train", train_results.metrics)
trainer.save_metrics("train", train_results.metrics)
trainer.save_state()

# Evaluate on validation set
metrics = trainer.evaluate(prepared_ds['validation'])
trainer.log_metrics("eval", metrics)
trainer.save_metrics("eval", metrics)