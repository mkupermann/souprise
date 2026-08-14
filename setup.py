#!/usr/bin/env python
"""Setup script for Souprise.

License: Apache-2.0
Copyright 2026 Michael Kupermann
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="souprise",
    version="0.1.0",
    author="Michael Kupermann",
    author_email="michael@kupermann.com",
    description="Offline RAG for Business Data: HDC Retrieval + LLM Generation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mkupermann/souprise",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Office/Business",
        "Topic :: Text Processing :: Linguistic",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10,<3.13",
    install_requires=[
        "soup-cli[mlx]>=0.73.1",
        "numpy>=1.26.0",
        "fastapi>=0.104.0",
        "uvicorn>=0.24.0",
        "typer>=0.9.0",
        "rich>=13.0.0",
        "pydantic>=2.0.0",
        "huggingface-hub>=0.16.0",
    ],
    extras_require={
        "retrieval": [
            "git+https://github.com/mkupermann/JuiceHDC.git@main#egg=cortex-hdc",
        ],
        "postgres": [
            "psycopg2-binary>=2.9.9",
            "sqlalchemy>=2.0.0",
        ],
        "dev": [
            "souprise[retrieval,postgres]",
            "pytest>=7.0",
            "ruff>=0.1.0",
            "mypy>=1.0",
            "pytest-cov>=4.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "souprise=souprise.cli.main:app",
        ],
    },
    include_package_data=True,
    package_data={
        "souprise": ["data/samples/*"],
    },
)
