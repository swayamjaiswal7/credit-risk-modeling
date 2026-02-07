"""
Setup file for credit-risk-model package
Allows installation with: pip install -e .
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_file = Path(__file__).parent / "README.md"
if readme_file.exists():
    with open(readme_file, encoding="utf-8") as f:
        long_description = f.read()
else:
    long_description = "Credit Risk Prediction System"

# Read requirements
requirements_file = Path(__file__).parent / "requirements.txt"
if requirements_file.exists():
    with open(requirements_file) as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]
else:
    requirements = []

setup(
    name="credit-risk-model",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Production ML system for credit risk prediction",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/credit-risk-model",
    packages=find_packages(exclude=["tests", "tests.*", "notebooks", "notebooks.*"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Financial and Insurance Industry",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "pylint>=2.17.0",
            "jupyter>=1.0.0",
            "jupyterlab>=4.0.0",
        ],
        "monitoring": [
            "prometheus-client>=0.17.0",
            "grafana-api>=1.0.3",
        ],
    },
    entry_points={
        "console_scripts": [
            "credit-risk-train=scripts.train_pipeline:main",
            "credit-risk-predict=scripts.predict_cli:main",
            "credit-risk-api=api.main:start_server",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["config/*.yaml", "config/*.yml"],
    },
    zip_safe=False,
    keywords=[
        "machine-learning",
        "credit-risk",
        "financial-modeling",
        "mlops",
        "production-ml",
        "fastapi",
        "docker",
    ],
    project_urls={
        "Bug Reports": "https://github.com/swayamjaiswal7/credit-risk-model/issues",
        "Source": "https://github.com/swayamjaiswal7/credit-risk-model",
        "Documentation": "https://github.com/swayamjaiswal7/credit-risk-model/wiki",
    },
)