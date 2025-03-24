from setuptools import setup, find_packages

setup(
    name="carbon_aware",
    version="0.1.0",
    description="Carbon-Aware Orchestrator implementation",
    author="TUBerlin",
    packages=find_packages(),
    install_requires=[
        "pandas",
        "matplotlib",
        "pyyaml",
        "grpcio",
        "protobuf",
        "pulp",
        "psutil",
    ],
    python_requires=">=3.8",
)