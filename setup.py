from setuptools import setup, find_packages

setup(
    name='presidential_model',
    version='0.1.0',
    description='A transformer-based model for token prediction using a reduced vocabulary.',
    author='Your Name',
    author_email='your.email@example.com',
    url='https://github.com/yourusername/presidential_model',  # Replace with your repository URL
    packages=find_packages(),
    install_requires=[
        'torch>=1.9.0',
        'transformers>=4.0.0',
        'pandas>=1.1.0',
        'numpy>=1.19.0',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.10',
)
