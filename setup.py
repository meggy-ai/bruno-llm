"""
Bruno-LLM - LLM provider implementations for bruno-core.

This package provides production-ready LLM provider implementations that
integrate seamlessly with the bruno-core framework.
"""

from pathlib import Path

from setuptools import find_packages, setup

# Read the contents of README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")

# Read version from __version__.py
version = {}
with open("bruno_llm/__version__.py") as fp:
    exec(fp.read(), version)

setup(
    name="bruno-llm",
    version=version["__version__"],
    author=version["__author__"],
    author_email=version["__email__"],
    description=version["__description__"],
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/meggy-ai/bruno-llm",
    project_urls={
        "Bug Tracker": "https://github.com/meggy-ai/bruno-llm/issues",
        "Documentation": "https://meggy-ai.github.io/bruno-llm/",
        "Source Code": "https://github.com/meggy-ai/bruno-llm",
        "Changelog": "https://github.com/meggy-ai/bruno-llm/blob/main/CHANGELOG.md",
    },
    packages=find_packages(exclude=["tests", "tests.*", "docs", "examples"]),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Typing :: Typed",
    ],
    python_requires=">=3.9",
    install_requires=[
        "bruno-core>=0.1.0",
        "httpx>=0.24.0",
        "aiohttp>=3.8.0",
        "pydantic>=2.0.0",
        "structlog>=23.1.0",
    ],
    extras_require={
        "openai": [
            "openai>=1.0.0",
            "tiktoken>=0.5.0",
        ],
        "dev": [
            "pytest>=7.4.0",
            "pytest-asyncio>=0.21.0",
            "pytest-cov>=4.1.0",
            "pytest-mock>=3.11.0",
            "mypy>=1.5.0",
            "ruff>=0.1.0",
            "pre-commit>=3.3.0",
            "black>=23.7.0",
        ],
        "docs": [
            "mkdocs>=1.5.0",
            "mkdocs-material>=9.1.0",
            "mkdocstrings[python]>=0.22.0",
        ],
    },
    entry_points={
        "bruno.llm_providers": [
            "ollama = bruno_llm.providers.ollama:OllamaProvider",
            "openai = bruno_llm.providers.openai:OpenAIProvider",
        ],
    },
    package_data={
        "bruno_llm": ["py.typed"],
    },
    include_package_data=True,
    zip_safe=False,
    license=version["__license__"],
    keywords="llm ai assistant openai ollama bruno chatbot language-model",
)
