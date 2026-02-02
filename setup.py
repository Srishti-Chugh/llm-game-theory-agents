from setuptools import setup, find_packages

setup(
    name="llm-game-theory-agents",
    version="0.1.0",
    description="LLM-based game theory agents simulation",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "numpy",
        "pandas",
        "matplotlib",
        "seaborn",
        "scipy",
        "tqdm",
        "python-dotenv",
        "openai",
        "transformers",
        "torch",
    ],
)
