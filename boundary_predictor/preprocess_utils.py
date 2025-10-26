"""
Preprocessing utilities for boundary predictor training.
"""
import langdetect


def is_foreign_language(text, base_language="en"):
    """Detect if the text is in a foreign language."""
    try:
        detected_lang = langdetect.detect(text)
        return detected_lang != base_language
    except:
        return False


def prepare_prompt_and_response(sample, dataset_name):
    """Prepare prompt and response for each sample in the dataset."""
    if dataset_name == "booksum":
        prompt = sample["prompt"]
        response = sample["completion"]
    elif dataset_name == "natural_questions":
        prompt = sample["prompt"]
        response = sample["completion"].strip().split("\n")[0]
    elif dataset_name in ["trivia_qa", "trivia_qa_unfiltered"]:
        context = sample["search_results"]["search_context"]
        context = "\n\n".join(context)
        prompt = context + "\n\n" + sample["question"]
        response = sample["answer"]["aliases"][0]
    elif dataset_name == "nvidia_ChatQA2_Long_SFT_data":
        prompt = sample["question"]
        response = sample["answer"]
    elif dataset_name == "nvidia_ChatQA2_Long_SFT_data_NarrativeQA_131072":
        prompt = sample["sub-paragraphs"] + "\n\n" + sample["question"]
        response = sample["answer"][0]
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    return prompt, response


def prepare_prompt_and_response_wrapper(example):
    """
    Wrapper function to prepare prompt and response
    for each sample in the dataset.
    """
    prompt, response = prepare_prompt_and_response(example, example["source"])
    return {"prompt": prompt, "response": response}