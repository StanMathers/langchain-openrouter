import re
from pathlib import Path

from setuptools import setup, find_packages

curren_dir = Path(__file__).parent


def generate_version(package_name: str) -> str:
    version = (Path("src") / package_name / "__init__.py").read_text()
    match = re.search("__version__ = ['\"]([^'\"]+)['\"]", version)
    if match:
        return match.group(1)
    raise RuntimeError("Unable to find version string.")


setup(
    name="langchain_openrouter",
    description="OpenRouter large language models for LangChain.",
    url="https://github.com/StanMathers/langchain-openrouter",
    version=generate_version("langchain_openrouter"),
    packages=find_packages("src"),
    package_dir={"": "src"},
    include_package_data=True,
    zip_safe=False,
    python_requires=">=3.10",
    install_requires=["requests", "langchain"],
    license="MIT",
    classifiers=[
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
)
