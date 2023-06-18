import numpy as np
import sklearn
import torch
from datasets import load_from_disk
from datasets import load_metric
from sklearn.metrics import auc, roc_curve
from transformers import Trainer
from transformers import TrainingArguments
from transformers import ViTFeatureExtractor
import pickle
from huggingface_hub import notebook_login
from datasets import load_from_disk
from datasets import load_metric
from transformers import AutoFeatureExtractor
from transformers import AutoModelForImageClassification, TrainingArguments, Trainer
import numpy as np
import torch
import pickle
# import torchvision.transforms
from transformers import ViTFeatureExtractor
from transformers import Swinv2Config, Swinv2Model
from transformers import ViTForImageClassification
import matplotlib.pyplot as plt
import tensorflow as tf

batch_size = 8

from torchvision.transforms import (
    CenterCrop,
    Compose,
    Normalize,
    RandomHorizontalFlip,
    RandomResizedCrop,
    Resize,
    ToTensor,
)


model_name_or_path = "swin2_v4_small"

feature_extractor = ViTFeatureExtractor.from_pretrained(model_name_or_path)


normalize = Normalize(mean=feature_extractor.image_mean, std=feature_extractor.image_std)
train_transforms = Compose(
        [
            RandomResizedCrop(feature_extractor.size),
            RandomHorizontalFlip(),
            ToTensor(),
            normalize,
        ]
    )

val_transforms = Compose(
        [
            Resize([feature_extractor.size['height'],feature_extractor.size['width']]),
            CenterCrop([feature_extractor.size['height'],feature_extractor.size['width']]),
            ToTensor(),
            normalize,
        ]
    )

def preprocess_train(example_batch):
    """Apply train_transforms across a batch."""
    example_batch["pixel_values"] = [
        train_transforms(image.convert("RGB")) for image in example_batch["image"]
    ]
    return example_batch

def preprocess_val(example_batch):
    """Apply val_transforms across a batch."""
    example_batch["pixel_values"] = [val_transforms(image.convert("RGB")) for image in example_batch["image"]]
    return example_batch

ds_test = load_from_disk('afterclear5')
ds_test.set_transform(preprocess_val)

metric = load_metric("accuracy")




model = AutoModelForImageClassification.from_pretrained( model_name_or_path,num_labels=2,ignore_mismatched_sizes=True)




model_name = model_name_or_path.split("/")[-1]


def compute_metrics(eval_pred):
    """Computes accuracy on a batch of predictions"""
    predictions = np.argmax(eval_pred.predictions, axis=1)
    return metric.compute(predictions=predictions, references=eval_pred.label_ids)


def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    labels = torch.tensor([example["labels"] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}

'''
def collate_fn(batch):
  #data collator
    return {
        'pixel_values': torch.stack([x['pixel_values'] for x in batch]),
        'labels': torch.tensor([x['label'] for x in batch])
    }
'''

args = TrainingArguments(
    output_dir="./swin_v3_sample",
    remove_unused_columns=False,
    evaluation_strategy = "steps",
    # save_strategy = "epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=batch_size,
    gradient_accumulation_steps=4,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=10,
    fp16=True,
    warmup_ratio=0.1,
    logging_steps=10,
    save_steps=100,
    eval_steps=100,
    load_best_model_at_end=True,
    save_total_limit=2,
    report_to='tensorboard',
    metric_for_best_model="accuracy",
    # push_to_hub=True,
)





trainer = Trainer(
    model,
    args,
    tokenizer=feature_extractor,
    compute_metrics=compute_metrics,
    data_collator=collate_fn,
)



model.eval()

resultaziz = trainer.predict(ds_test)

print(resultaziz)
print(sum(np.nanargmax(resultaziz.predictions,axis = -1)))
score = tf.math.softmax(resultaziz.predictions,axis = -1)
print((score))
scores =[]
i = 0
for s in score:

    if resultaziz.label_ids[i] == 0:
        score = 1 - s[0]
    else:
        score = s[1]
    i +=1
    scores.append(score)




def roc_curve_plot(fpr,tpr,roc_auc):
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' %roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()
# np.nanargmax(resultaziz.predictions,axis = -1)
fpr, tpr, _ = roc_curve(resultaziz.label_ids,scores)
roc_auc = auc(fpr, tpr)
roc_curve_plot(fpr,tpr,roc_auc)
#
#
#



# # create a binary pickle file
#f = open("Swin1_tiny_celeba.pkl","wb")

# write the python object (dict) to pickle file
#pickle.dump(resultaziz,f)

# close file
#f.close()
#


s_fmr, tpr, thr = sklearn.metrics.ranking.roc_curve(resultaziz.label_ids,scores)
euc = 1 - sklearn.metrics.auc(s_fmr, tpr)
s_fnmr = 1 - tpr

eer_index = np.argmin(np.abs(s_fmr - s_fnmr))
eer = (s_fmr[eer_index] + s_fnmr[eer_index]) / 2
print(eer,'eer')
print(euc,'euc')
# # ret = plt.plot(s_fmr, s_fnmr,label = "goozoo")
# # plt.legend()
#
# res = 0
# i = 0
# NPAIS = len((resultaziz.label_ids))-sum(resultaziz.label_ids)
#
# for i in range(67170):
#     if resultaziz.label_ids[i] == 0:
#         if np.nanargmax(resultaziz.predictions,axis = -1)[i]== 0:
#             res+=1
#
# print(1-(1/NPAIS)*res, 'APCER')
#
# i =0
# res = 0
# for i in range(67170):
#     if resultaziz.label_ids[i] == 1:
#         if np.nanargmax(resultaziz.predictions,axis = -1)[i]== 0:
#             res+=1
#
# print(res/sum(resultaziz.label_ids), 'BPCER')
#
#
#
# plt.show()
#
