import setuptools

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()


__version__ = "0.1.0"

REPO_NAME = "self-healing-network"
AUTHOR_USER_NAME = "logicsame"
SRC_REPO = "Bioneural"
AUTHOR_EMAIL = "useforprofessional@gmail.com"


setuptools.setup(
    name=SRC_REPO,
    version=__version__,
    author=AUTHOR_USER_NAME,
    author_email=AUTHOR_EMAIL,
    description="A Bio logical self healing neuron layer. in the latest update add contextual repair mechanism, contextual noise injection",
    long_description=long_description,
    long_description_content="text/markdown",
    url=f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}",
    project_urls={
        "Bug Tracker": f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}/issues",
    },
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    install_requires=[
        "torch>=1.7.0",
        "numpy>=1.19.2",
        'pandas',
        'matplotlib',
        'seaborn'
    ],
    python_requires=">=3.7"
)