import pytest

from langchain_openrouter import OpenRouterLLM  # noqa: F401, E402, E902 # type: ignore

from .conftest import OPENROUTER_API_KEY


def prepare() -> OpenRouterLLM:
    return OpenRouterLLM(api_key=OPENROUTER_API_KEY)


def teardown(data: OpenRouterLLM):
    del data


@pytest.fixture
def setup():
    data = prepare()
    yield data
    teardown(data)
