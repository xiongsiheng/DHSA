from enum import Enum

from .metrics import (
    qa_f1_score,
    rouge_score
)


# Base model names
class Model(Enum):
    gemma_2_2b = "google/gemma-2-2b-it"
    gemma_3_1b = "google/gemma-3-1b-it"


# Attention implementations
class AttnImplementation(Enum):
    sdpa = "sdpa"
    eager = "eager"
    flash_attention_2 = "flash_attention_2"


# Sparse attention method names
class SparseAttnMethod(Enum):
    full = "full"
    h2o = "h2o"
    pyramidkv = "pyramidkv"
    streamingllm = "streamingllm"
    slidingwindow = "slidingwindow"
    topk = "topk"
    blocksparse = "blocksparse"
    dhsa = "dhsa"


# Pooling methods
class PoolingMethod(Enum):
    maxpool = "maxpool"
    avgpool = "avgpool"
    avgpool_norm = "avgpool_norm"


# Sampling methods for LongBench test data
class SampleMethod(Enum):
    random = "random"
    firstk = "firstk"


# Model paths for each model
MODEL_PATHS = {
    Model.gemma_2_2b: 'google/gemma-2-2b-it',  # copied from https://huggingface.co/google/gemma-2-2b-it
    Model.gemma_3_1b: 'google/gemma-3-1b-it'  # copied from https://huggingface.co/google/gemma-3-1b-it
}


# Pretrained lens for each model
PRETRAINED_LENS = {
    Model.gemma_2_2b: 8192,
    Model.gemma_3_1b: 32768
}


# Maximum prompt lengths for each model
MODEL_MAX_PROMPT_LENGTHS = {
    Model.gemma_2_2b: 7950,
    Model.gemma_3_1b: 31500
}


# Period tokens ('.': end of sentence) for each model
PERIOD_TOKENS = {
    Model.gemma_2_2b: [235265],
    Model.gemma_3_1b: [236761]
}


# Model providers for each model
MODEL_PROVIDERS = {
    Model.gemma_2_2b: "Gemma2",
    Model.gemma_3_1b: "Gemma3"
}


# Needles used in Needle-in-a-haystack Test (can be changed independently)
NEEDLE = "\nThe best thing to do in San Francisco is eat a sandwich and sit in Dolores Park on a sunny day.\n"
RETRIEVAL_QUESTION = "The best thing to do in San Francisco is: "
RETRIEVAL_ANSWER = "eat a sandwich and sit in Dolores Park on a sunny day."
PROMPT_TEMPLATE = (
    "<|im_start|> This is a very long story book: <book> {context} </book>."
    "\n Based on the content of the book, Question: {retrieval_question}\nAnswer:"
)
INSTRUCTION_TEMPLATE = (
    "Based on the content of the book, Question: {retrieval_question}\nAnswer:"
)

# Directory for data and results in Needle-in-a-haystack Test
DATA_DIR = "data/PaulGrahamEssays"
RESULTS_DIR = "results_needle"


# Default context lengths and number of iterations for latency test
CONTEXT_LENGTHS_LATENCY_TEST = [1024, 2048, 4096, 8192, 16384, 32768]  # 1K to 32K tokens
NUM_ITERATIONS_LATENCY_TEST = 3



# LongBench dataset configurations
LongBench_DATASETS = [
    "narrativeqa", "qasper", "multifieldqa_en",
    "hotpotqa", "2wikimqa", "musique",
    "qmsum", "triviaqa",
    "gov_report", "multi_news", "samsum"
]

# Maximum output lengths for each dataset
LongBench_DATASET_MAX_LENGTHS = {
    "narrativeqa": 128,
    "qasper": 128,
    "multifieldqa_en": 64,
    "hotpotqa": 32,
    "2wikimqa": 32,
    "musique": 32,
    "qmsum": 512,
    "triviaqa": 32,
    "gov_report": 512,
    "multi_news": 512,
    "samsum": 128,
}

# Prompts for each dataset
LongBench_TASK_PROMPTS = {
    "narrativeqa": (
        "You are given a story, which can be either a novel or a movie script, and a question. "
        "Answer the question as concisely as you can, using a single phrase if possible. "
        "Do not provide any explanation.\n\n"
        "Story: {context}\n\n"
        "Now, answer the question based on the story as concisely as you can, using a single phrase if possible. "
        "Do not provide any explanation.\n\n"
        "Question: {input}\n\nAnswer:"
    ),
    "qasper": (
        "You are given a scientific article and a question. Answer the question as concisely as you can, "
        "using a single phrase or sentence if possible. If the question cannot be answered based on the "
        "information in the article, write \"unanswerable\". If the question is a yes/no question, "
        "answer \"yes\", \"no\", or \"unanswerable\". Do not provide any explanation.\n\n"
        "Article: {context}\n\n"
        "Answer the question based on the above article as concisely as you can, using a single phrase or sentence if possible. "
        "If the question cannot be answered based on the information in the article, write \"unanswerable\". "
        "If the question is a yes/no question, answer \"yes\", \"no\", or \"unanswerable\". "
        "Do not provide any explanation.\n\n"
        "Question: {input}\n\nAnswer:"
    ),
    "multifieldqa_en": (
        "Read the following text and answer briefly.\n\n{context}\n\n"
        "Now, answer the following question based on the above text, only give me the answer "
        "and do not output any other words.\n\n"
        "Question: {input}\nAnswer:"
    ),
    "hotpotqa": (
        "Answer the question based on the given passages. Only give me the answer "
        "and do not output any other words.\n\n"
        "The following are given passages.\n{context}\n\n"
        "Answer the question based on the given passages. Only give me the answer "
        "and do not output any other words.\n\n"
        "Question: {input}\nAnswer:"
    ),
    "2wikimqa": (
        "Answer the question based on the given passages. Only give me the answer "
        "and do not output any other words.\n\n"
        "The following are given passages.\n{context}\n\n"
        "Answer the question based on the given passages. Only give me the answer "
        "and do not output any other words.\n\n"
        "Question: {input}\nAnswer:"
    ),
    "musique": (
        "Answer the question based on the given passages. Only give me the answer "
        "and do not output any other words.\n\n"
        "The following are given passages.\n{context}\n\n"
        "Answer the question based on the given passages. Only give me the answer "
        "and do not output any other words.\n\n"
        "Question: {input}\nAnswer:"
    ),
    "gov_report": (
        "You are given a report by a government agency. Write a one-page summary of the report.\n\n"
        "Report:\n{context}\n\n"
        "Now, write a one-page summary of the report.\n\n"
        "Summary:"
    ),
    "qmsum": (
        "You are given a meeting transcript and a query containing a question or instruction. "
        "Answer the query in one or more sentences.\n\n"
        "Transcript:\n{context}\n\n"
        "Now, answer the query based on the above meeting transcript in one or more sentences.\n\n"
        "Query: {input}\nAnswer:"
    ),
    "multi_news": (
        "You are given several news passages. Write a one-page summary of all news.\n\n"
        "News:\n{context}\n\n"
        "Now, write a one-page summary of all the news.\n\n"
        "Summary:"
    ),
    "triviaqa": (
        "Answer the question based on the given passage. Only give me the answer "
        "and do not output any other words. The following are some examples.\n\n"
        "{context}{input}"
    ),
    "samsum": (
        "Summarize the dialogue into a few short sentences. The following are some examples.\n\n"
        "{context}\n\n{input}"
    )
}

# Instruction prompts for each dataset
LongBench_INSTRUCTION_PROMPTS = {
    "narrativeqa": (
        "Question: {input}\n\nAnswer:"
    ),
    "qasper": (
        "Question: {input}\n\nAnswer:"
    ),
    "multifieldqa_en": (
        "Question: {input}\nAnswer:"
    ),
    "hotpotqa": (
        "Question: {input}\nAnswer:"
    ),
    "2wikimqa": (
        "Question: {input}\nAnswer:"
    ),
    "musique": (
        "Question: {input}\nAnswer:"
    ),
    "qmsum": (
        "Query: {input}\nAnswer:"
    ),
    "triviaqa": (
        "{input}"
    ),
    "gov_report": (
        "Now, write a one-page summary of the report.\n\n"
        "Summary:"
    ),
    "multi_news": (
        "Now, write a one-page summary of all the news.\n\n"
        "Summary:"
    ),
    "samsum": (
        "Summary:"
    )
}


# Metric functions for each dataset
LongBench_DATASET_METRICS = {
    "narrativeqa": qa_f1_score,
    "qasper": qa_f1_score,
    "multifieldqa_en": qa_f1_score,
    "hotpotqa": qa_f1_score,
    "2wikimqa": qa_f1_score,
    "musique": qa_f1_score,
    "triviaqa": qa_f1_score,
    "gov_report": rouge_score,
    "qmsum": rouge_score,
    "multi_news": rouge_score,
    "samsum": rouge_score,
}

# Datasets that require preprocessing
LongBench_DATASETS_REQUIRING_PREPROCESSING = {"triviaqa", "samsum"}


# Directory for data and results in LongBench Test
LongBench_DATA_DIR = 'data/LongBench'
LongBench_RESULTS_DIR = 'results_longbench'


# Directory for Boundary Predictor Training
Boundary_TRAINING_DIR = 'boundary_predictor_training'