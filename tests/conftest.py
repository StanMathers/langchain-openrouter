import os
import sys

import environ  # type: ignore

# OpenRouterLLM can not be imported by default
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from langchain_openrouter import OpenRouterLLM  # noqa: F401, E402 # type: ignore

env = environ.Env()
environ.Env.read_env("test.env")

OPENROUTER_API_KEY = env("OPENROUTER_API_KEY")
