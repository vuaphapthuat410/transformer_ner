from datasets import load_dataset
from sklearn.metrics import classification_report, f1_score
from transformers import BertTokenizerFast, Trainer, TrainingArguments
from transformers.trainer_utils import IntervalStrategy

from models import BertForTokenClassification

task = "ner" # Should be one of "ner", "pos" or "chunk"
model_checkpoint = "microsoft/Multilingual-MiniLM-L12-H384"
batch_size = 16

from datasets import load_dataset, load_metric

datasets = load_dataset(f"{task}_script.py")

def remove_segment(example):
  re_tokens = []
  re_ner_tags = []

  for i, x in enumerate(example['tokens']):
    num = x.count("_")
    if num == 0:
      re_tokens.append(x)
      re_ner_tags.append(example['ner_tags'][i])
    else:
      sub_str = x.split("_")
      for j, y in enumerate(sub_str):
        re_tokens.append(y)
      if example['ner_tags'][i] in [1,3,5,7]:
        re_ner_tags.append(example['ner_tags'][i])
        re_ner_tags.extend([example['ner_tags'][i]+1]*(num))
      else:
        re_ner_tags.extend([example['ner_tags'][i]]*(num+1))

  example['tokens'] = re_tokens
  example['ner_tags'] = re_ner_tags
  return example

no_seg_datasets = datasets.map(remove_segment, batched=False)

label_list = no_seg_datasets["train"].features[f"{task}_tags"].feature.names

train_dataset, test_dataset = no_seg_datasets['train'], no_seg_datasets['validation']

label2id = {
    "B-LOC": 5,
    "B-MISC": 7,
    "B-ORG": 3,
    "B-PER": 1,
    "I-LOC": 6,
    "I-MISC": 8,
    "I-ORG": 4,
    "I-PER": 2,
    "O": 0
  }
id2label = {
    0: 'O',
    1: 'B-PER',
    2: 'I-PER',
    3: 'B-ORG',
    4: 'I-ORG',
    5: 'B-LOC',
    6: 'I-LOC',
    7: 'B-MISC',
    8: 'I-MISC'
  }

from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer, AutoConfig
config = AutoConfig.from_pretrained(model_checkpoint, label2id=label2id, id2label=id2label, num_labels=len(label_list), classifier_dropout=0.2)
model = BertForTokenClassification.from_pretrained(model_checkpoint, config = config)

from transformers import XLMRobertaTokenizerFast
    
tokenizer = XLMRobertaTokenizerFast.from_pretrained(model_checkpoint, do_lower_case=False)

from transformers import DataCollatorForTokenClassification

data_collator = DataCollatorForTokenClassification(tokenizer)

from transformers import EarlyStoppingCallback, TrainerControl, TrainerCallback

def tokenize(batch):
    result = {
        'label_ids': [],
        'input_ids': [],
        'token_type_ids': [],
    }
    max_length = 512

    for tokens, label in zip(batch['tokens'], batch['label_ids']):
        tokenids = tokenizer(tokens, add_special_tokens=False)

        token_ids = []
        label_ids = []
        for ids, lab in zip(tokenids['input_ids'], label):
            if len(ids) > 1 and lab % 2 == 1:
                token_ids.extend(ids)
                chunk = [lab + 1] * len(ids)
                chunk[0] = lab
                label_ids.extend(chunk)
            else:
                token_ids.extend(ids)
                chunk = [lab] * len(ids)
                label_ids.extend(chunk)

        token_type_ids = tokenizer.create_token_type_ids_from_sequences(token_ids)
        token_ids = tokenizer.build_inputs_with_special_tokens(token_ids)
        label_ids.insert(0, 0)
        label_ids.append(0)
        result['input_ids'].append(token_ids)
        result['label_ids'].append(label_ids)
        result['token_type_ids'].append(token_type_ids)

    result = tokenizer.pad(result, padding='longest', max_length=max_length, return_attention_mask=True)
    for i in range(len(result['input_ids'])):
        diff = len(result['input_ids'][i]) - len(result['label_ids'][i])
        result['label_ids'][i] += [0] * diff
    return result


train_dataset = train_dataset.remove_columns(['id'])
train_dataset = train_dataset.rename_column('ner_tags', 'label_ids')
test_dataset = test_dataset.remove_columns(['id'])
test_dataset = test_dataset.rename_column('ner_tags', 'label_ids')

train_dataset = train_dataset.map(tokenize, batched=True, batch_size=len(train_dataset))
test_dataset = test_dataset.map(tokenize, batched=True, batch_size=len(test_dataset))
train_dataset.set_format('torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'label_ids'])
test_dataset.set_format('torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'label_ids'])


def compute_metrics(pred):
    labels = pred.label_ids.flatten()
    preds = pred.predictions.flatten()
    f1 = f1_score(labels, preds, average='macro')
    print(classification_report(labels, preds))
    return {
        'f1': f1
    }


training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=15,
    per_device_train_batch_size=64,
    per_device_eval_batch_size=64,
    # weight_decay=0.01,
    # save_strategy=IntervalStrategy.EPOCH,
    logging_dir='./logs',
    evaluation_strategy ='steps',
    warmup_steps = 100,
    eval_steps = 100,
    save_steps=100,
    save_total_limit = 15,
    metric_for_best_model = 'eval_loss',
    load_best_model_at_end=True,
    learning_rate=1e-5,
)

trainer = Trainer(
    model=model,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    # callbacks = [EarlyStoppingCallback(early_stopping_patience=3)],
)

trainer.train()

print(trainer.evaluate())

trainer.save_model("./")
