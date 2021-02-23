import setuptools
import os
from pathlib import Path

with open("README.md", "r") as fh:
    long_description = fh.read()
# long_description = "Long description"
setuptools.setup(
    name="wildpython",
    version="0.1.0",
    author="Conor Wild",
    author_email="conorwild@gmail.com",
    description="A package for misc and resuable code bits",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="",
    packages=['wildpython'],
    include_package_data=False,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        'numpy', 'pandas', 'matplotlib', 'statsmodels', 'plotly', 'pingouin', 'mergedeep'
    ],
    entry_points='''
        [console_scripts]
    ''',
    options={}
)
