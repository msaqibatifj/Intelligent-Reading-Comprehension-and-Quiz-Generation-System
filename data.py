from datasets import load_dataset

# 1. Download the dataset from the Hub
dataset = load_dataset("ehovy/race", "all")

# 2. Save it to a local directory
dataset.save_to_disk("./local_race_dataset")

print("Dataset saved locally to './local_race_dataset'")