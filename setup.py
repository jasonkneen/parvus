from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="parvus",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A quantum-inspired data compression system for efficient storage and retrieval",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/parvus",
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
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: System :: Archiving :: Compression",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "gpu": ["torch>=1.9.0", "faiss-gpu>=1.7.0"],
        "dev": [
            "pytest>=6.0",
            "black>=21.0",
            "isort>=5.0",
            "flake8>=3.9",
            "mypy>=0.900",
        ],
    }
)