from setuptools import setup, find_packages

setup(
    name="turingtrader",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=[
        "ib_insync",
        "pandas",
        "numpy",
        "matplotlib",
        "yfinance",
    ],
    extras_require={
        "dev": [
            "pytest",
            "pytest-cov",
            "black",
            "flake8",
        ],
    },
    description="AI driven algorithmic trader for S&P500 options",
    author="mazharm",
    license="MIT",
)