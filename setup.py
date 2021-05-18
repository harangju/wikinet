import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="wikinet",
    version="0.1.2",
    author="Harang Ju",
    author_email="harangju@gmail.com",
    description="Network of wikipedia articles",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/harangju/wikinet",
    project_urls={
        "Bug Tracker": "https://github.com/harangju/wikinet/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3.7",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages=["wikinet"],
    python_requires=">=3.7",
    install_requires=[
        'numpy',
        'scipy',
        'pandas>1.0.0',
        'networkx',
        'gensim',
        'python-Levenshtein', # gensim warning
        'bctpy>=0.5.2',
        'mwparserfromhell',
        'sklearn',
        'dionysus<=2.0.7', # weird dependency issue with python>3.7
        'cpnet<=0.0.6',
        'pytest'
    ]
)
