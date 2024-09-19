from setuptools import setup, find_packages

setup(
    name='ffa_net_project',
    version='0.1',
    packages=find_packages('src'),
    package_dir={'': 'src'},
    install_requires=[
        'numpy',
        'torch',
        'scikit-learn',
        'matplotlib',
        'pandas',
    ],
    entry_points={
        'console_scripts': [
            'ffa_net=src.main:main',
        ],
    },
)
