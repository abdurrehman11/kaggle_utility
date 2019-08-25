from setuptools import setup, find_packages

setup(
    name='kaggle_utility',
    version='1.1',
    author='Abdur Rehman Nadeem',
    description='A utility for kaggle machine learning projects',
    license='MIT',
    url='https://github.com/abdurrehman11/kaggle_utility',
    packages=find_packages(),
    install_requires=[
        'scikit-learn==0.20.3',
        'pandas==0.24.2',
        'numpy==1.16.3',
        'xgboost==0.90',
        ],
)