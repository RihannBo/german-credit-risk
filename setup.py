from setuptools import setup, find_packages


def get_requirements(file_path: str) -> list[str]:
    """Read and clean requirements from file."""
    with open(file_path) as f:
        return [
            line.strip()
            for line in f
            if line.strip() and not line.startswith("#") and line.strip() != "-e ."
        ]


setup(
    name="credit_risk_analytics",
    version="0.0.1",
    author="ReihanBo",
    author_email="reihaneh.boustani@gmail.com",
    
    # 🔥 Important for src layout
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    
    install_requires=get_requirements("requirements.txt"),
    
    python_requires=">=3.9",
)