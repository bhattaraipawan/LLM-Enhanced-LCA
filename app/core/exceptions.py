"""Application-level exceptions shared by controllers and core integrations."""


class LlamaUnavailableError(RuntimeError):
    """Raised internally when the optional local Llama runtime cannot be used."""

    prefix = "Llama is not available"

    def __init__(self, reason: str | None = None) -> None:
        reason = (reason or "").strip()
        if not reason:
            message = self.prefix
        elif reason.startswith(self.prefix):
            message = reason
        else:
            message = f"{self.prefix}: {reason}"
        self.reason = message
        super().__init__(message)


class OpenLCAUnavailableError(RuntimeError):
    """Raised when the configured openLCA runtime cannot be reached."""


class CalculationError(RuntimeError):
    """Raised when an LCA calculation cannot be completed."""


class InvalidInputError(ValueError):
    """Raised when a controller receives invalid user input."""
