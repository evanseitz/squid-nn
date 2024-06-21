from setuptools import setup, find_packages

setup(
    name="squid-nn",
    version="0.4.2",
    author="Evan Seitz",
    author_email="evan.e.seitz@gmail.com",
    packages=find_packages(),
    description = "SQUID is a TensorFlow package to interpret sequence-based deep learning models for regulatory genomics data with domain-specific surrogate models.",
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires=">=3.7.2",
    install_requires=[
    	'numpy',
	'matplotlib>=3.2.0',
	'pandas',
        'tqdm',
    ],
)