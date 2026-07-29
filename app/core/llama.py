"""Lazy, in-process access to an optional local Llama model.

This module intentionally imports only the Python standard library. PyTorch and
Transformers are imported on the first generation request, after a supported
GPU has been detected. Consequently, importing the application and rendering
its GUI never depends on the model, Hugging Face access, or GPU availability.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import importlib
import json
import math
import threading
from typing import Any

from .exceptions import LlamaUnavailableError


DEFAULT_SYSTEM_PROMPT = (
    "You are a strict JSON generator. Return ONLY one JSON object matching the "
    "keys and value types requested by the user. Do not return prose, markdown, "
    "or code fences."
)


class LlamaState(str, Enum):
    """Lifecycle state of a :class:`LlamaEngine`."""

    UNINITIALIZED = "uninitialized"
    LOADING = "loading"
    READY = "ready"
    UNAVAILABLE = "unavailable"
    ERROR = "error"


@dataclass(frozen=True, slots=True)
class LlamaConfig:
    """Configuration supplied by the application when it builds the engine."""

    model_name: str = "meta-llama/Llama-3.1-8B-Instruct"
    allow_mps: bool = True
    local_files_only: bool = False
    trust_remote_code: bool = False
    hf_token: str | None = field(default=None, repr=False)
    default_max_new_tokens: int = 256
    max_new_tokens_limit: int = 1024
    system_prompt: str = DEFAULT_SYSTEM_PROMPT

    def __post_init__(self) -> None:
        if not self.model_name.strip():
            raise ValueError("model_name must not be blank")
        if self.default_max_new_tokens < 1:
            raise ValueError("default_max_new_tokens must be positive")
        if self.max_new_tokens_limit < 1:
            raise ValueError("max_new_tokens_limit must be positive")


@dataclass(frozen=True, slots=True)
class LlamaGenerationResult:
    """Typed outcome returned for both successful and unavailable generations."""

    available: bool
    raw_output: str = ""
    result: dict[str, float] = field(default_factory=dict)
    message: str | None = None

    @classmethod
    def unavailable(cls, message: str) -> "LlamaGenerationResult":
        error = LlamaUnavailableError(message)
        return cls(available=False, message=str(error))


@dataclass(slots=True)
class _Runtime:
    torch: Any
    tokenizer: Any
    model: Any
    device: str


def extract_json_block(text: str) -> str:
    """Return the first balanced JSON object in *text*.

    Brace characters inside JSON strings are ignored. If no balanced object is
    present, the input is returned so a subsequent ``json.loads`` can fail in
    the same way as the original implementation.
    """

    stripped = text.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        return stripped

    start: int | None = None
    depth = 0
    in_string = False
    escaped = False

    for index, character in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
            continue

        if character == '"':
            in_string = True
        elif character == "{":
            if depth == 0:
                start = index
            depth += 1
        elif character == "}" and depth:
            depth -= 1
            if depth == 0 and start is not None:
                return text[start : index + 1]

    return text


def parse_json_object(raw: str) -> dict[str, Any]:
    """Parse the first JSON object in *raw*, returning an empty dict on error."""

    try:
        parsed = json.loads(extract_json_block(raw))
    except (TypeError, json.JSONDecodeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def parse_structured_floats(raw: str) -> dict[str, float]:
    """Extract numeric-convertible top-level values from a generated object."""

    result: dict[str, float] = {}
    for key, value in parse_json_object(raw).items():
        try:
            number = float(value)
        except (TypeError, ValueError, OverflowError):
            continue
        if math.isfinite(number):
            result[key] = number
    return result


class LlamaEngine:
    """Thread-safe lazy owner of a single local model and tokenizer."""

    def __init__(
        self,
        config: LlamaConfig | None = None,
        **config_fields: Any,
    ) -> None:
        if config is not None and config_fields:
            raise ValueError("pass either config or keyword configuration, not both")
        self.config = config if config is not None else LlamaConfig(**config_fields)
        self._condition = threading.Condition(threading.RLock())
        self._inference_lock = threading.Lock()
        self._state = LlamaState.UNINITIALIZED
        self._runtime: _Runtime | None = None
        self._message: str | None = None

    def status(self) -> dict[str, Any]:
        """Return dict-like state without importing dependencies or loading a model."""

        with self._condition:
            runtime = self._runtime
            return {
                "state": self._state.value,
                "available": self._state is LlamaState.READY,
                "device": runtime.device if runtime is not None else None,
                "message": self._message,
                "model_name": self.config.model_name,
            }

    def load(self) -> dict[str, Any]:
        """Explicitly request loading, returning status instead of raising."""

        try:
            self._ensure_loaded()
        except LlamaUnavailableError:
            pass
        return self.status()

    def generate(
        self,
        prompt: str,
        max_new_tokens: int | None = None,
    ) -> LlamaGenerationResult:
        """Generate strict JSON, or return a typed unavailable outcome."""

        if not isinstance(prompt, str):
            raise TypeError("prompt must be a string")

        try:
            self._ensure_loaded()
        except LlamaUnavailableError as exc:
            return LlamaGenerationResult.unavailable(str(exc))

        token_limit = self._normalize_token_limit(max_new_tokens)

        # Loading and inference use separate locks: callers may all wait for the
        # same load, while generation remains serialized to protect GPU memory.
        with self._inference_lock:
            with self._condition:
                if self._state is not LlamaState.READY or self._runtime is None:
                    return LlamaGenerationResult.unavailable(
                        self._message or "the model runtime stopped"
                    )
                runtime = self._runtime

            try:
                raw_output = self._generate(runtime, prompt, token_limit)
            except Exception as exc:
                reason = self._describe_runtime_error(
                    exc, runtime.torch, "while generating a response"
                )
                self._try_empty_gpu_cache(runtime.torch, runtime.device)
                error = LlamaUnavailableError(reason)
                self._record_failure(LlamaState.ERROR, str(error))
                return LlamaGenerationResult.unavailable(str(error))

        return LlamaGenerationResult(
            available=True,
            raw_output=raw_output,
            result=parse_structured_floats(raw_output),
        )

    def generate_raw_output(
        self,
        prompt: str,
        max_new_tokens: int | None = None,
    ) -> str:
        """Compatibility helper returning raw text or raising unavailability."""

        outcome = self.generate(prompt, max_new_tokens)
        if not outcome.available:
            raise LlamaUnavailableError(outcome.message)
        return outcome.raw_output

    def _ensure_loaded(self) -> None:
        with self._condition:
            while self._state is LlamaState.LOADING:
                self._condition.wait()

            if self._state is LlamaState.READY:
                return
            if self._state in {LlamaState.UNAVAILABLE, LlamaState.ERROR}:
                raise LlamaUnavailableError(self._message)

            self._state = LlamaState.LOADING
            self._message = None

        try:
            runtime = self._build_runtime()
        except LlamaUnavailableError as exc:
            self._record_failure(LlamaState.UNAVAILABLE, str(exc))
            raise
        except Exception as exc:
            error = LlamaUnavailableError(f"model initialization failed ({self._detail(exc)})")
            self._record_failure(LlamaState.ERROR, str(error))
            raise error from exc

        with self._condition:
            self._runtime = runtime
            self._state = LlamaState.READY
            self._message = None
            self._condition.notify_all()

    def _build_runtime(self) -> _Runtime:
        try:
            torch = importlib.import_module("torch")
        except Exception as exc:
            raise LlamaUnavailableError(
                f"PyTorch could not be imported ({self._detail(exc)})"
            ) from exc

        device = self._select_gpu(torch)

        try:
            transformers = importlib.import_module("transformers")
            auto_tokenizer = transformers.AutoTokenizer
            auto_model = transformers.AutoModelForCausalLM
        except Exception as exc:
            raise LlamaUnavailableError(
                f"Transformers could not be imported ({self._detail(exc)})"
            ) from exc

        common_kwargs: dict[str, Any] = {
            "local_files_only": self.config.local_files_only,
            "trust_remote_code": self.config.trust_remote_code,
        }
        if self.config.hf_token:
            common_kwargs["token"] = self.config.hf_token

        try:
            tokenizer = auto_tokenizer.from_pretrained(
                self.config.model_name,
                use_fast=True,
                **common_kwargs,
            )

            model_kwargs = dict(common_kwargs)
            if device == "cuda":
                model_kwargs["device_map"] = {"": "cuda"}
                model_kwargs["torch_dtype"] = self._cuda_dtype(torch)
            else:
                # MPS is selected explicitly after construction; no automatic
                # device map is allowed to silently offload inference to CPU.
                model_kwargs["torch_dtype"] = getattr(torch, "float16", None)
                if model_kwargs["torch_dtype"] is None:
                    model_kwargs.pop("torch_dtype")

            model = auto_model.from_pretrained(self.config.model_name, **model_kwargs)
            if device == "mps":
                moved_model = model.to("mps")
                if moved_model is not None:
                    model = moved_model
            if hasattr(model, "eval"):
                model.eval()
        except Exception as exc:
            self._try_empty_gpu_cache(torch, device)
            raise LlamaUnavailableError(
                self._describe_runtime_error(exc, torch, "while loading the model")
            ) from exc

        if (
            getattr(tokenizer, "pad_token_id", None) is None
            and getattr(tokenizer, "eos_token_id", None) is not None
        ):
            tokenizer.pad_token = tokenizer.eos_token

        return _Runtime(torch=torch, tokenizer=tokenizer, model=model, device=device)

    def _select_gpu(self, torch: Any) -> str:
        try:
            cuda = getattr(torch, "cuda", None)
            if cuda is not None and bool(cuda.is_available()):
                return "cuda"

            if self.config.allow_mps:
                backends = getattr(torch, "backends", None)
                mps = getattr(backends, "mps", None)
                if mps is not None and bool(mps.is_available()):
                    return "mps"
        except Exception as exc:
            raise LlamaUnavailableError(
                f"GPU detection failed ({self._detail(exc)})"
            ) from exc

        supported = "CUDA or Apple MPS" if self.config.allow_mps else "CUDA"
        raise LlamaUnavailableError(f"no supported GPU was detected ({supported} required)")

    @staticmethod
    def _cuda_dtype(torch: Any) -> Any:
        cuda = getattr(torch, "cuda", None)
        supports_bfloat16 = getattr(cuda, "is_bf16_supported", None)
        if callable(supports_bfloat16):
            try:
                if supports_bfloat16():
                    return torch.bfloat16
            except Exception:
                pass
        return torch.float16

    def _generate(
        self,
        runtime: _Runtime,
        prompt: str,
        max_new_tokens: int,
    ) -> str:
        messages = [
            {"role": "system", "content": self.config.system_prompt},
            {"role": "user", "content": prompt},
        ]
        rendered_prompt = runtime.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        model_inputs = runtime.tokenizer([rendered_prompt], return_tensors="pt")
        model_inputs = model_inputs.to(runtime.device)
        prompt_length = self._prompt_length(model_inputs)

        inference_mode = getattr(runtime.torch, "inference_mode", None)
        if not callable(inference_mode):
            inference_mode = runtime.torch.no_grad

        with inference_mode():
            generated_ids = runtime.model.generate(
                **model_inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=runtime.tokenizer.pad_token_id,
                eos_token_id=runtime.tokenizer.eos_token_id,
            )

        output_ids = generated_ids[0][prompt_length:]
        return runtime.tokenizer.decode(output_ids, skip_special_tokens=True).strip()

    @staticmethod
    def _prompt_length(model_inputs: Any) -> int:
        input_ids = model_inputs["input_ids"]
        shape = getattr(input_ids, "shape", None)
        if shape is not None:
            return int(shape[-1])
        return len(input_ids[0])

    def _normalize_token_limit(self, requested: int | None) -> int:
        if requested is None:
            requested = self.config.default_max_new_tokens
        if not isinstance(requested, int):
            raise TypeError("max_new_tokens must be an integer")
        return max(1, min(requested, self.config.max_new_tokens_limit))

    def _record_failure(self, state: LlamaState, message: str) -> None:
        error = LlamaUnavailableError(message)
        with self._condition:
            self._runtime = None
            self._state = state
            self._message = str(error)
            self._condition.notify_all()

    @classmethod
    def _describe_runtime_error(cls, exc: Exception, torch: Any, phase: str) -> str:
        if cls._is_out_of_memory(exc, torch):
            return f"the GPU ran out of memory {phase}"

        detail = cls._detail(exc)
        lower_detail = detail.lower()
        if any(term in lower_detail for term in ("gated", "401", "403", "unauthorized")):
            return f"model authorization failed {phase} ({detail})"
        if "accelerate" in lower_detail:
            return f"a required model dependency is missing {phase} ({detail})"
        return f"the model failed {phase} ({detail})"

    @staticmethod
    def _is_out_of_memory(exc: Exception, torch: Any) -> bool:
        cuda = getattr(torch, "cuda", None)
        oom_type = getattr(cuda, "OutOfMemoryError", None)
        if isinstance(oom_type, type) and isinstance(exc, oom_type):
            return True
        return "out of memory" in str(exc).lower()

    @staticmethod
    def _try_empty_gpu_cache(torch: Any, device: str) -> None:
        if device != "cuda":
            return
        try:
            empty_cache = getattr(torch.cuda, "empty_cache", None)
            if callable(empty_cache):
                empty_cache()
        except Exception:
            pass

    @staticmethod
    def _detail(exc: Exception) -> str:
        detail = " ".join(str(exc).split()) or type(exc).__name__
        return detail[:300]
