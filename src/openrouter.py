import json
from typing import List, Mapping, Any, Optional

import requests
from langchain.llms.base import LLM
from langchain.callbacks.manager import CallbackManager


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
    """

    URL = "https://openrouter.ai/api/v1/chat/completions"

    @classmethod
    def init(
        cls,
        api_key: str,
        llm_type: Optional[str] = "gpt-3.5-turbo",
        model: Optional[str] = "openai/gpt-3.5-turbo",
        **kwargs: Any,
    ):
        """`LLM` requires to set class variables to access them in `_call` method.

        `init` is chosen since it acts as a constructor for the class variables in this case.
        """
        cls.api_key = api_key
        cls.llm_type = llm_type
        cls.model = model

        cls.n = kwargs.get("n", 1)
        cls.max_tokens = kwargs.get("max_tokens", None)
        cls.temperature = kwargs.get("temperature", None)
        cls.top_k = kwargs.get("top_k", None)
        cls.top_p = kwargs.get("top_p", 1.0)
        cls.presence_penalty = kwargs.get("presence_penalty", 0.0)
        cls.frequency_penalty = kwargs.get("frequency_penalty", 0.0)
        cls.repetition_penalty = kwargs.get("repetition_penalty", 1.0)
        cls.min_p = kwargs.get("min_p", 0.0)
        cls.top_a = kwargs.get("top_a", 0.0)
        cls.seed = kwargs.get("seed", None)
        cls.logit_bias = kwargs.get("logit_bias", None)

    @property
    def _llm_type(self) -> str:
        return self.llm_type

    def _call(
        self,
        prompt: str,
        stop: Optional[str] = None,
        callback_manager: Optional[CallbackManager] = None,
    ) -> List[str]:
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
