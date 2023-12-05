from setuptools import setup, find_packages

setup(
    name='chatlm',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'openai>=1',
        'pydantic>=2',
        'ml-collections'
    ],
)
