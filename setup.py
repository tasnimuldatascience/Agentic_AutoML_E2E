from setuptools import setup, find_packages

setup(
    name='agentic-automl',
    version='0.1.0',
    author='Tasnimul Hasan',
    author_email='tasnimuldatascience@gmail.com',
    description='Agentic AutoML: An AutoML framework with SHAP and LLM explanations',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/tasnimuldatascience/agentic_automl',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'streamlit',
        'scikit-learn',
        'matplotlib',
        'shap',
        'xgboost',
        'lightgbm',
        'optuna',
        'openai==0.28',
        'pandas',
        'numpy',
        'python-dotenv',
        'seaborn',
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.8',
    ],
    python_requires='>=3.8',
    entry_points={
        'console_scripts': [
            'agentic-automl=app:main',
        ],
    },
)
