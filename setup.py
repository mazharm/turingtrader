from setuptools import setup, find_packages

setup(
    name="turing_trader",
    version="0.1.0",
    description="AI driven algorithmic trader using Interactive Brokers API",
    author="mazharm",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "ibapi>=9.76.1",
        "numpy>=1.19.0",
        "pandas>=1.0.0",
        "matplotlib>=3.3.0",
        "seaborn>=0.11.0",
        "scipy>=1.5.0",
        "yfinance>=0.1.63",
        "python-dotenv>=0.15.0",
        "requests>=2.25.0",
        "loguru>=0.5.0",
    ],
    python_requires=">=3.8",
)