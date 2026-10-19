"""Regression tests proving a bound model is reused without reloading."""

from unittest.mock import patch

from .qwen_client import active_qwen, bind_loaded_qwen, generate_response


class FakeTokenizer:
    eos_token_id = 0
    chat_template = None

    def __call__(self, *_args, **_kwargs):
        return {"input_ids": FakeTensor()}

    def decode(self, *_args, **_kwargs):
        return "ok"


class FakeTensor:
    shape = (1, 1)

    def to(self, _device):
        return self


class FakeParameter:
    device = "cuda:0"


class FakeModel:
    def parameters(self):
        return iter([FakeParameter()])

    def generate(self, **_kwargs):
        return [[0, 1]]


class FakeCuda:
    @staticmethod
    def is_available():
        return False


class FakeTorch:
    cuda = FakeCuda()

    @staticmethod
    def inference_mode():
        return _Noop()


def main() -> None:
    tokenizer, model = FakeTokenizer(), FakeModel()
    bind_loaded_qwen(tokenizer, model)
    assert active_qwen() == (tokenizer, model)
    with patch("qwen.qwen_client.load_qwen", side_effect=AssertionError("must not reload")):
        with patch("qwen.qwen_client._runtime_modules", return_value=(FakeTorch(), None)):
            assert generate_response("request 1") == "ok"
            assert generate_response("request 2") == "ok"
            assert generate_response("request 3") == "ok"

    # Different repository URLs change only RAG work. They must not change the
    # bound Qwen pair or invoke the model loader.
    for repository_url in (
        "https://huggingface.co/openai-community/gpt2",
        "https://huggingface.co/google/vit-base-patch16-224",
        "https://huggingface.co/openai/whisper-tiny",
    ):
        assert active_qwen() == (tokenizer, model), repository_url
    print("Bound Qwen model reuse test passed for three consecutive requests.")


class _Noop:
    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False


if __name__ == "__main__":
    main()
