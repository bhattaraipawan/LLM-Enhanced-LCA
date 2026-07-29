from contextlib import nullcontext
import threading
import unittest
from unittest.mock import patch

from app.core.llama import (
    LlamaConfig,
    LlamaEngine,
    LlamaGenerationResult,
    LlamaState,
    extract_json_block,
    parse_json_object,
    parse_structured_floats,
)


class _FakeMPS:
    def __init__(self, available=False):
        self._available = available

    def is_available(self):
        return self._available


class _FakeCuda:
    class OutOfMemoryError(RuntimeError):
        pass

    def __init__(self, available=False, bf16=False):
        self._available = available
        self._bf16 = bf16
        self.empty_cache_calls = 0

    def is_available(self):
        return self._available

    def is_bf16_supported(self):
        return self._bf16

    def empty_cache(self):
        self.empty_cache_calls += 1


class _FakeTorch:
    float16 = "float16"
    bfloat16 = "bfloat16"

    def __init__(self, cuda=False, mps=False, bf16=False):
        self.cuda = _FakeCuda(cuda, bf16)
        self.backends = type("Backends", (), {"mps": _FakeMPS(mps)})()

    @staticmethod
    def inference_mode():
        return nullcontext()


class _FakeBatch(dict):
    def __init__(self):
        super().__init__(input_ids=[[10, 11]])
        self.target_device = None

    def to(self, device):
        self.target_device = device
        return self


class _FakeTokenizer:
    def __init__(self, raw_output):
        self.raw_output = raw_output
        self.pad_token_id = None
        self.eos_token_id = 2
        self.eos_token = "</s>"
        self.pad_token = None
        self.last_batch = None

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        return "rendered prompt"

    def __call__(self, texts, return_tensors):
        self.last_batch = _FakeBatch()
        return self.last_batch

    def decode(self, output_ids, skip_special_tokens):
        return self.raw_output


class _FakeModel:
    def __init__(self):
        self.to_calls = []
        self.generate_calls = []
        self.eval_called = False

    def to(self, device):
        self.to_calls.append(device)
        return self

    def eval(self):
        self.eval_called = True

    def generate(self, **kwargs):
        self.generate_calls.append(kwargs)
        return [[10, 11, 20, 21]]


def _fake_transformers(
    raw_output='prefix {"score": 2, "label": "cement"} suffix',
    load_error=None,
    tokenizer_started=None,
    tokenizer_release=None,
):
    counters = {"tokenizer": 0, "model": 0}
    tokenizer = _FakeTokenizer(raw_output)
    model = _FakeModel()

    class AutoTokenizer:
        @staticmethod
        def from_pretrained(model_name, **kwargs):
            counters["tokenizer"] += 1
            if tokenizer_started is not None:
                tokenizer_started.set()
            if tokenizer_release is not None:
                tokenizer_release.wait(timeout=2)
            return tokenizer

    class AutoModel:
        @staticmethod
        def from_pretrained(model_name, **kwargs):
            counters["model"] += 1
            if load_error is not None:
                raise load_error
            model.load_kwargs = kwargs
            return model

    module = type(
        "FakeTransformers",
        (),
        {
            "AutoTokenizer": AutoTokenizer,
            "AutoModelForCausalLM": AutoModel,
        },
    )()
    return module, tokenizer, model, counters


def _module_importer(torch, transformers=None):
    def import_module(name):
        if name == "torch":
            if isinstance(torch, BaseException):
                raise torch
            return torch
        if name == "transformers":
            if isinstance(transformers, BaseException):
                raise transformers
            return transformers
        raise AssertionError(f"unexpected import: {name}")

    return import_module


class LlamaPureFunctionTests(unittest.TestCase):
    def test_extracts_json_while_ignoring_braces_inside_strings(self):
        raw = 'prose {"note": "a } brace", "value": 3} trailing'
        self.assertEqual(
            extract_json_block(raw),
            '{"note": "a } brace", "value": 3}',
        )

    def test_parsers_preserve_object_and_filter_non_numeric_values(self):
        raw = 'answer: {"score": "2.5", "name": "sand", "count": 3}'
        self.assertEqual(
            parse_json_object(raw),
            {"score": "2.5", "name": "sand", "count": 3},
        )
        self.assertEqual(parse_structured_floats(raw), {"score": 2.5, "count": 3.0})
        self.assertEqual(parse_structured_floats("not json"), {})


class LlamaEngineTests(unittest.TestCase):
    def test_construction_does_not_import_or_load_optional_dependencies(self):
        with patch("app.core.llama.importlib.import_module") as importer:
            engine = LlamaEngine(LlamaConfig())
            self.assertEqual(engine.status()["state"], LlamaState.UNINITIALIZED.value)
            importer.assert_not_called()

    def test_constructor_accepts_direct_keyword_configuration(self):
        engine = LlamaEngine(model_name="local/test-model", allow_mps=False)
        self.assertEqual(engine.config.model_name, "local/test-model")
        self.assertFalse(engine.config.allow_mps)
        self.assertEqual(engine.status()["model_name"], "local/test-model")

    def test_missing_torch_returns_typed_unavailable_result(self):
        engine = LlamaEngine(LlamaConfig())
        importer = _module_importer(ModuleNotFoundError("No module named 'torch'"))
        with patch("app.core.llama.importlib.import_module", side_effect=importer):
            outcome = engine.generate("Return JSON")

        self.assertIsInstance(outcome, LlamaGenerationResult)
        self.assertFalse(outcome.available)
        self.assertTrue(outcome.message.startswith("Llama is not available"))
        self.assertIn("PyTorch", outcome.message)

    def test_no_gpu_never_imports_transformers_or_attempts_cpu_fallback(self):
        torch = _FakeTorch(cuda=False, mps=False)
        imported = []

        def importer(name):
            imported.append(name)
            if name == "torch":
                return torch
            raise AssertionError("Transformers must not be imported without a GPU")

        engine = LlamaEngine(LlamaConfig())
        with patch("app.core.llama.importlib.import_module", side_effect=importer):
            first = engine.generate("Return JSON")
            second = engine.generate("Return JSON")

        self.assertFalse(first.available)
        self.assertFalse(second.available)
        self.assertTrue(first.message.startswith("Llama is not available"))
        self.assertIn("no supported GPU", first.message)
        self.assertEqual(imported, ["torch"])
        self.assertEqual(engine.status()["state"], LlamaState.UNAVAILABLE.value)

    def test_cuda_load_and_generation_are_lazy_and_cached(self):
        torch = _FakeTorch(cuda=True, bf16=True)
        transformers, tokenizer, model, counters = _fake_transformers()
        engine = LlamaEngine(LlamaConfig(max_new_tokens_limit=128))
        importer = _module_importer(torch, transformers)

        with patch("app.core.llama.importlib.import_module", side_effect=importer):
            first = engine.generate("Return score", max_new_tokens=999)
            second = engine.generate("Return score")

        self.assertTrue(first.available)
        self.assertEqual(first.result, {"score": 2.0})
        self.assertEqual(first.raw_output, 'prefix {"score": 2, "label": "cement"} suffix')
        self.assertTrue(second.available)
        self.assertEqual(counters, {"tokenizer": 1, "model": 1})
        self.assertEqual(model.load_kwargs["device_map"], {"": "cuda"})
        self.assertEqual(model.load_kwargs["torch_dtype"], "bfloat16")
        self.assertEqual(model.generate_calls[0]["max_new_tokens"], 128)
        self.assertEqual(tokenizer.last_batch.target_device, "cuda")
        self.assertEqual(engine.status()["state"], LlamaState.READY.value)
        self.assertEqual(engine.status()["device"], "cuda")

    def test_mps_is_supported_only_when_enabled(self):
        torch = _FakeTorch(cuda=False, mps=True)
        transformers, tokenizer, model, _ = _fake_transformers()
        engine = LlamaEngine(LlamaConfig(allow_mps=True))

        with patch(
            "app.core.llama.importlib.import_module",
            side_effect=_module_importer(torch, transformers),
        ):
            outcome = engine.generate("Return JSON")

        self.assertTrue(outcome.available)
        self.assertEqual(model.to_calls, ["mps"])
        self.assertEqual(tokenizer.last_batch.target_device, "mps")
        self.assertNotIn("device_map", model.load_kwargs)

    def test_gated_model_error_becomes_unavailable(self):
        torch = _FakeTorch(cuda=True)
        transformers, _, _, _ = _fake_transformers(
            load_error=PermissionError("401 gated repository")
        )
        engine = LlamaEngine(LlamaConfig())

        with patch(
            "app.core.llama.importlib.import_module",
            side_effect=_module_importer(torch, transformers),
        ):
            outcome = engine.generate("Return JSON")

        self.assertFalse(outcome.available)
        self.assertTrue(outcome.message.startswith("Llama is not available"))
        self.assertIn("authorization failed", outcome.message)

    def test_model_oom_is_caught_and_cache_is_cleared(self):
        torch = _FakeTorch(cuda=True)
        transformers, _, _, _ = _fake_transformers(
            load_error=torch.cuda.OutOfMemoryError("CUDA out of memory")
        )
        engine = LlamaEngine(LlamaConfig())

        with patch(
            "app.core.llama.importlib.import_module",
            side_effect=_module_importer(torch, transformers),
        ):
            outcome = engine.generate("Return JSON")

        self.assertFalse(outcome.available)
        self.assertIn("ran out of memory", outcome.message)
        self.assertEqual(torch.cuda.empty_cache_calls, 1)

    def test_concurrent_requests_share_one_model_load(self):
        load_started = threading.Event()
        release_load = threading.Event()
        torch = _FakeTorch(cuda=True)
        transformers, _, _, counters = _fake_transformers(
            tokenizer_started=load_started,
            tokenizer_release=release_load,
        )
        engine = LlamaEngine(LlamaConfig())
        outcomes = []

        def invoke():
            outcomes.append(engine.generate("Return JSON"))

        with patch(
            "app.core.llama.importlib.import_module",
            side_effect=_module_importer(torch, transformers),
        ):
            first = threading.Thread(target=invoke)
            second = threading.Thread(target=invoke)
            first.start()
            self.assertTrue(load_started.wait(timeout=1))
            second.start()
            release_load.set()
            first.join(timeout=2)
            second.join(timeout=2)

        self.assertEqual(len(outcomes), 2)
        self.assertTrue(all(outcome.available for outcome in outcomes))
        self.assertEqual(counters, {"tokenizer": 1, "model": 1})


if __name__ == "__main__":
    unittest.main()
