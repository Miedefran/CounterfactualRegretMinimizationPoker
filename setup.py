from setuptools import setup, find_packages

setup(
    name='poker-cfr',
    version='0.1.0',
    packages=find_packages(),
    python_requires='>=3.7',
    install_requires=[
        'pandas>=1.3.0',
    ],
    extras_require={
        'gui': [
            'PyQt6>=6.0.0',
            'Flask>=2.0.0',
            'requests>=2.28.0',
        ],
    },
)

