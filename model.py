from transformers import AutoTokenizer
from dataset_preparation import load_dataset

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

dataset = load_dataset()


def tokenize_function(dataset):
    return tokenizer(dataset["content"], padding="max_length", truncation=True)


tokenized_datasets = dataset.map(tokenize_function, batched=True)
