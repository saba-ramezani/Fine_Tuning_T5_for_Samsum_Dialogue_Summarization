# Fine-Tuning T5 for Samsum Dialogue Summarization

This project demonstrates **fine-tuning T5** on the **Samsum dialogue summarization dataset**. It also benchmarks T5 against BART on a sample summarization task from the CNN/DailyMail dataset.  

---

## Features

- Load and explore the **Samsum dataset**
- Benchmark **T5** and **BART** on a sample summarization task
- Fine-tune **T5-small** for dialogue summarization
- Use Hugging Face **Trainer** API for training
- Generate summaries from custom dialogues using the fine-tuned model

---

##  Installation

Install required libraries:

```bash
pip install -U transformers
pip install -U accelerate
pip install -U datasets
pip install -U bertviz
pip install -U umap-learn
pip install -U sentencepiece
pip install -U urllib3
pip install py7zr
```

## Benchmarking T5 vs BART
Example of benchmarking on CNN-DailyMail:

```
from transformers import pipeline
import torch

device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
summary = {}

# T5-small fine-tuned on CNN
pipe = pipeline('summarization', model='ubikpt/t5-small-finetuned-cnn', device=device)
output = pipe(dataset[0]['article'])
summary['t5-small'] = output[0]['summary_text']

# BART-large-CNN
pipe = pipeline('summarization', model='facebook/bart-large-cnn', device=device)
output = pipe(dataset[0]['article'])
summary['bart-large'] = output[0]['summary_text']

print(summary)

t5-small
Harry Potter star Daniel Radcliffe says he has no plans to fritter his cash away . The actor has filmed a TV movie about author Rudyard Kipling

bart-large
Harry Potter star Daniel Radcliffe turns 18 on Monday. He gains access to a reported £20 million ($41.1 million) fortune. Radcliffe's earnings from the first five Potter films have been held in a trust fund. Details of how he'll mark his landmark birthday are under wraps.
```
Observation:

T5 provides a shorter summary (~60M parameters)

BART produces a more detailed summary (~406M parameters)

## Samsum Dialogue Dataset
### Load the dataset:

```
from datasets import load_dataset

samsum = load_dataset("knkarthick/samsum")
```
### Dataset structure:

```
DatasetDict({
    train: Dataset({
        features: ['id', 'dialogue', 'summary'],
        num_rows: 14731
    })
    validation: Dataset({
        features: ['id', 'dialogue', 'summary'],
        num_rows: 818
    })
    test: Dataset({
        features: ['id', 'dialogue', 'summary'],
        num_rows: 819
    })
})
```
## Data Analysis
Example: dialogue and summary length distributions:

```
import pandas as pd

dialogue_len = [len(d.split()) for d in samsum['train']['dialogue']]
summary_len = [len(s.split()) for s in samsum['train']['summary']]

data = pd.DataFrame([dialogue_len, summary_len]).T
data.columns = ['Dialogue Length', 'Summary Length']
data.hist(figsize=(10,3))
```
<img width="839" height="297" alt="image" src="https://github.com/user-attachments/assets/57e2e220-5fec-40df-86f5-e0f7d5032d41" />


## Preprocessing & Tokenization

```
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_ckpt = 't5-small'
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
model = AutoModelForSeq2SeqLM.from_pretrained(model_ckpt).to(device)
```
## Fine-Tuning T5
```
from transformers import DataCollatorForSeq2Seq, TrainingArguments, Trainer

# Data collator for seq2seq tasks
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

# Training arguments
args = TrainingArguments(
    output_dir="train_dir",
    num_train_epochs=2,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    eval_strategy='epoch',
    save_strategy='epoch',
    weight_decay=0.01,
    learning_rate=2e-5,
    gradient_accumulation_steps=500,
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=args,
    tokenizer=tokenizer,
    data_collator=data_collator,
    train_dataset=samsum['train'],
    eval_dataset=samsum['validation']
)

# Train the model
trainer.train()
```
## Summarizing Custom Dialogue
After fine-tuning, you can summarize any custom dialogue:

```
from transformers import pipeline

pipe = pipeline(
    'summarization', 
    model='/content/drive/MyDrive/llm_finetuning_transformers/Dialogue_Summarization/t5-small-dialogue-summarization-model', 
    device=device
)

custom_dialogue = """
Laxmi Kant: what work you planning to give Tom?
Juli: i was hoping to send him on a business trip first.
Laxmi Kant: cool. is there any suitable work for him?
Juli: he did excellent in last quarter. i will assign new project, once he is back.
"""

output = pipe(custom_dialogue)
print(output)
```
Example output:
```
[{'summary_text': 'Laxmi Kant: i was hoping to send him on a business trip first. I will assign new project once he is back.'}]
```

## Results & Observations
T5-small fine-tuned on Samsum produces coherent dialogue summaries.

T5 provides concise summaries suitable for dialogue data.

BART tends to produce more detailed summaries but has significantly more parameters.

## Usage
Clone this repo and install dependencies.

Load the Samsum dataset.

Fine-tune T5-small using Trainer.

Use the trained model with the Hugging Face pipeline for summarizing dialogues.

## Future Work
Experiment with T5-base or T5-large for higher quality summaries.

Try BART or Pegasus for dialogue summarization.

Add evaluation metrics: ROUGE, BLEU, BERTScore.

Deploy the model as an API or web application for real-time summarization.

## License
This project is for research and educational purposes only.
Not intended for direct clinical or production use.
