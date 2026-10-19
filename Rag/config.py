"""Configuration for the small Hugging Face RAG component."""

HF_API_URL = "https://huggingface.co/api"
HF_RAW_URL = "https://huggingface.co"
GITHUB_API_URL = "https://api.github.com"
REQUEST_TIMEOUT_SECONDS = 30
MAX_FILES_PER_REPOSITORY = 40
MAX_TOTAL_DOCUMENT_CHARS = 500_000

RELEVANT_FILE_NAMES = {
    "README.md",
    "README",
    "requirements.txt",
    "requirements-dev.txt",
    "pyproject.toml",
    "setup.py",
    "setup.cfg",
    "environment.yml",
    "environment.yaml",
    "config.json",
    "model_index.json",
    "generation_config.json",
    "tokenizer_config.json",
    "processor_config.json",
    "preprocessor_config.json",
    "quantization_config.json",
    "adapter_config.json",
}
RELEVANT_DOCUMENT_SUFFIXES = (".md", ".rst", ".toml", ".yaml", ".yml", ".txt")
RELEVANT_SOURCE_DIRECTORIES = {"examples", "example", "scripts", "script", "src", "demo", "demos", "inference"}
RELEVANT_SOURCE_NAMES = {
    "pipeline.py",
    "inference.py",
    "predict.py",
    "modeling.py",
    "configuration.py",
    "processing.py",
    "tokenization.py",
    "train.py",
}
MAX_DOCUMENT_CHARS = 100_000
DEFAULT_TOP_K = 5
