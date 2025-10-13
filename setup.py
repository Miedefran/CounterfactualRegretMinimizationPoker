from setuptools import setup, find_packages

setup(
    name='poker-cfr',
    version='0.1.0',
    packages=find_packages(),
    python_requires='>=3.7',
    install_requires=[
        'pandas',
    ],
)

