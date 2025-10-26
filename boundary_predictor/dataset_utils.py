"""
Dataset utilities for boundary predictor training.
"""
import datasets
from preprocess_utils import prepare_prompt_and_response_wrapper


# Centralize all dataset information in a list of dictionaries.
DATASET_CONFIGS = [
    {
        "path": "togethercomputer/Long-Data-Collections",
        "data_files": "fine-tune/booksum.jsonl.zst",
        "source_name": "booksum",
    },
    {
        "path": "togethercomputer/Long-Data-Collections",
        "data_files": "fine-tune/natural_questions_10_200_docs.jsonl.zst",
        "source_name": "natural_questions",
    },
    {
        "path": "mandarjoshi/trivia_qa",
        "name": "rc",
        "source_name": "trivia_qa"
    },
    {
        "path": "mandarjoshi/trivia_qa",
        "name": "unfiltered",
        "source_name": "trivia_qa_unfiltered"
    },
    {
        "path": "nvidia/ChatQA2-Long-SFT-data",
        "name": "long_sft",
        "source_name": "nvidia_ChatQA2_Long_SFT_data"
    },
    {
        "path": "nvidia/ChatQA2-Long-SFT-data",
        "name": "NarrativeQA_131072",
        "source_name": "nvidia_ChatQA2_Long_SFT_data_NarrativeQA_131072",
    },
]


def train_test_split(
    dataset: datasets.Dataset,
    test_size_ratio: float = 0.2,
    seed: int = 42
):
    """
    Split a dataset into training and test sets.

    Args:
        dataset: The dataset to split.
        test_size_ratio: The ratio of the test set to the total dataset.
        seed: The seed for splitting the dataset.

    Returns:
        A tuple of training and test datasets.
    """
    # 1. Shuffle the dataset for randomness
    #    Use a seed for reproducibility.
    shuffled_dataset = dataset.shuffle(seed=seed)

    # 2. Determine the split size
    num_samples = len(shuffled_dataset)
    split_index = int((1 - test_size_ratio) * num_samples)

    # 3. Select the indices for each split
    train_indices = range(split_index)
    test_indices = range(split_index, num_samples)

    # 4. Create the final train and test datasets
    train_dataset = shuffled_dataset.select(train_indices)
    test_dataset = shuffled_dataset.select(test_indices)

    return train_dataset, test_dataset



def prepare_datasets(
    num_samples_train: int = 10000,
    num_samples_val: int = 100,
    test_size_ratio: float = 0.2,
    seed: int = 42
):
  """
  Prepare and combine multiple datasets for training by iterating through a central configuration list.

  Args:
      num_samples_train: Number of samples to select from the training split.
      num_samples_val: Number of samples to select from the validation split.
      test_size_ratio: Ratio of the validation split to the total dataset.
      seed: Seed for splitting the datasets.

  Returns:
      A tuple of training and validation datasets.
  """
  train_datasets, val_datasets = [], []
  final_columns = ["prompt", "response", "uid", "source"]

  # Loop through the configuration list instead of hardcoding each dataset.
  for config in DATASET_CONFIGS:
      # Load dataset using parameters from the config dictionary.
      # .get() gracefully handles optional keys like 'name' or 'data_files'.
      dataset = datasets.load_dataset(
          path=config["path"],
          name=config.get("name"),
          data_files=config.get("data_files"),
          split="train",
      )

      train_dataset, val_dataset = train_test_split(dataset, test_size_ratio=test_size_ratio, seed=seed)

      # Sample and process the training split
      train_ds = train_dataset.select(range(min(num_samples_train, len(train_dataset))))
      train_ds = train_ds.map(lambda ex, idx: {"uid": idx, "source": config["source_name"]}, with_indices=True)
      train_ds = train_ds.map(prepare_prompt_and_response_wrapper)
      train_ds = train_ds.remove_columns([col for col in train_ds.column_names if col not in final_columns])
      train_datasets.append(train_ds)

      # Sample and process the validation split
      val_ds = val_dataset.select(range(min(num_samples_val, len(val_dataset))))
      val_ds = val_ds.map(lambda ex, idx: {"uid": idx, "source": config["source_name"]}, with_indices=True)
      val_ds = val_ds.map(prepare_prompt_and_response_wrapper)
      val_ds = val_ds.remove_columns([col for col in val_ds.column_names if col not in final_columns])
      val_datasets.append(val_ds)

  # Concatenate all datasets at the end.
  return datasets.concatenate_datasets(train_datasets), datasets.concatenate_datasets(val_datasets)