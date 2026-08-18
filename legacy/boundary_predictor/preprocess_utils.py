"""
Preprocessing utilities for boundary predictor training.
"""
import langdetect
from langdetect.lang_detect_exception import LangDetectException


def is_foreign_language(text, base_language="en"):
    """Detect if the text is in a foreign language."""
    try:
        detected_lang = langdetect.detect(text)
        return detected_lang != base_language
    except LangDetectException:
        return False


def _join_text(value, separator="\n\n"):
    """Normalize string or list-valued dataset fields to text."""
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        return separator.join(str(item) for item in value)
    return str(value)


def _first_text(value, field_name):
    """Return a scalar answer from a string or a non-empty list."""
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)) and value:
        return str(value[0])
    raise ValueError(f"Expected a non-empty string or list in {field_name!r}")


def prepare_prompt_and_response(sample, dataset_name):
    """Prepare prompt and response for each sample in the dataset."""
    if dataset_name == "booksum":
        prompt = sample["prompt"]
        response = sample["completion"]
    elif dataset_name == "natural_questions":
        prompt = sample["prompt"]
        response = sample["completion"].strip().split("\n")[0]
    elif dataset_name in ["trivia_qa", "trivia_qa_unfiltered"]:
        context = _join_text(sample["search_results"]["search_context"])
        prompt = context + "\n\n" + sample["question"]
        response = _first_text(sample["answer"]["aliases"], "answer.aliases")
    elif dataset_name == "nvidia_ChatQA2_Long_SFT_data":
        prompt = _join_text(sample["question"])
        response = _first_text(sample["answer"], "answer")
    elif dataset_name == "nvidia_ChatQA2_Long_SFT_data_NarrativeQA_131072":
        prompt = _join_text(sample["sub-paragraphs"]) + "\n\n" + _join_text(sample["question"])
        response = _first_text(sample["answer"], "answer")
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
