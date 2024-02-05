import json
from typing import Any, List, Mapping, Optional

import requests
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM


class OpenRouterLLM(LLM):
    """OpenRouter large language models.

    To use, you should have the `OpenRouter` account and generated `API` keys.

    `llm_type` is the valid OpenRouter LLM type. Before using any of them, you should
    check the OpenRouter LLM types documentation with `model` parameter.

    `model` is the valid OpenRouter LLM model name. Before using any of them, you
    should check the OpenRouter LLM models documentation.

    `https://openrouter.ai/docs#models`

    `**kwargs` are the valid parameters for the OpenRouter LLM parameters. Before
    using any of them, you should check the OpenRouter LLM parameters documentation.

    `https://openrouter.ai/docs#llm-parameters`


    `LLM` requires to set class variables to access them in `_call` method.

    """

    URL = "https://openrouter.ai/api/v1/chat/completions"

    api_key: str = ""
    llm_type: str = "gpt-3.5-turbo"
    model: str = "openai/gpt-3.5-turbo"
    n: int = 1
    max_tokens: Optional[int] | None = None
    temperature: Optional[float] | None = None
    top_k: Optional[int] | None = None
    top_p: Optional[float] = 1.0
    presence_penalty: Optional[float] = 0.0
    frequency_penalty: Optional[float] = 0.0
    repetition_penalty: Optional[float] = 1.0
    min_p: Optional[float] = 0.0
    top_a: Optional[float] = 0.0
    seed: Optional[int] | None = None
    logit_bias: Optional[Mapping[str, float]] | None = None

    @property
    def _llm_type(self) -> str:
        return self.llm_type

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        params = {
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_k": self.top_k,
            "top_p": self.top_p,
            "presence_penalty": self.presence_penalty,
            "frequency_penalty": self.frequency_penalty,
            "repetition_penalty": self.repetition_penalty,
            "min_p": self.min_p,
            "top_a": self.top_a,
            "seed": self.seed,
            "logit_bias": self.logit_bias,
        }

        data = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
        }
        response = requests.post(
            self.URL, headers=headers, data=json.dumps(data), params=params
        )
        response.raise_for_status()
        response_data = response.json()
        return response_data["choices"][0]["message"]["content"]

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {
            "api_key": self.api_key,
            "llm_type": self.llm_type,
            "model": self.model,
            "n": self.n,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_k": self.top_k,
            "top_p": self.top_p,
            "presence_penalty": self.presence_penalty,
            "frequency_penalty": self.frequency_penalty,
            "repetition_penalty": self.repetition_penalty,
            "min_p": self.min_p,
            "top_a": self.top_a,
            "seed": self.seed,
            "logit_bias": self.logit_bias,
        }
