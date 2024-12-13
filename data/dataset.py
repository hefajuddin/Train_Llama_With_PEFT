from datasets import Dataset
from data.context import sample_texts

# # Convert the text into a Dataset object
# data = {"text": sample_texts}
# dataset = Dataset.from_dict(data)

from datasets import load_dataset
dataset = load_dataset("text", data_files={"train": "train.txt", "validation": "validation.txt"})

