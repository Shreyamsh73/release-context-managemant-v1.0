"""Context Gateway: a self-hosted memory layer for stateless LLM APIs."""

import os

# Must be set before LiteLLM is imported anywhere. Stops LiteLLM from fetching a
# model cost map from the internet at import time.
os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")

__version__ = "0.1.0"

__all__ = ["__version__"]
