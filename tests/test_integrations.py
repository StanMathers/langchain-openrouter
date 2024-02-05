from .fixtures import setup  # noqa: F401


def test_api_key(setup):
    assert setup.api_key != ""


def test_response(setup):
    llm = setup
    assert llm("Say: `Hello World!`") == "Hello World!"
