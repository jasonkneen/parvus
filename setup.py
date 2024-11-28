from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="parvus",
    version="0.1.1",
    author="Jason Kneen",
    author_email="jason@biouncingfish.com",
    description="A quantum-inspired data compression and semantic search system",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jasonkneen/parvus",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: System :: Archiving :: Compression",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.19.0",
        "scipy>=1.6.0",
        "scikit-learn>=0.24.0",
        "sentence-transformers>=2.2.0",
        "torch>=1.8.0",
        "faiss-cpu>=1.7.0",
        "flask>=2.0.0",
        "streamlit>=1.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "black>=22.0.0",
            "isort>=5.0.0",
            "mypy>=0.900",
            "flake8>=3.9.0",
        ],
        "gpu": ["faiss-gpu>=1.7.0"],
    },
    entry_points={
        "console_scripts": [
            "parvus=parvus.__main__:main",
        ],
    },
)
