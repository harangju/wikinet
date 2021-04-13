import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="wikinet",
    version="0.0.10",
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
    package_dir={"": "wikinet"},
    packages=setuptools.find_packages(where="wikinet"),
    python_requires="==3.7", # need to make >=
    install_requires=[
        'cython',
        'jupyter',
        'numpy',
        'scipy',
        'pandas>1.0.0',
        'networkx',
        'gensim',
        'python-Levenshtein', # gensim warning
        'mpmath',
        'sphinx',
        'nbconvert==5.6.1', # https://github.com/ipython-contrib/jupyter_contrib_nbextensions/issues/1529#issuecomment-695057809
        'plotly',
#        'plotly-orca', # saving plotly
        'psutil', # saving plotly
        'leidenalg',
        'python-igraph',
        'bctpy>=0.5.2',
        'mwparserfromhell',
        'sklearn',
        'dionysus<=2.0.7', # weird dependency issue with python>3.7
        'pybind11',
        'cpnet<=0.0.6',
        'powerlaw',
        'sphinx_rtd_theme',
        'cufflinks',
        'dill',
        'rpy2',
        'build',
        'twine',
        'pytest'
    ]
)
