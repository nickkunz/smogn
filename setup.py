from setuptools import setup

setup(
    
    name = 'smogn',
    version = '0.1.1',
    description = 'A Python implementation of Synthetic Minority Over-Sampling Technique for Regression with Gaussian Noise (SMOGN)',
    long_description = open('README.md').read(),
    long_description_content_type = "text/markdown",
    author = 'Nick Kunz',
    author_email = 'nick.kunz@columbia.edu',
    url = 'https://github.com/nickkunz/smogn',
    classifiers = [

        'Programming Language :: Python :: 3',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Education',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)'
        ],
    
    keywords = [
        
        'smote',
        'over-sampling',
        'synthetic data',
        'imbalanced data',
        'pre-processing',
        'regression'
    ],
    
    packages = ['smogn'],
    include_package_data = True,
    extras_require={
        "dev": ["pytest>=7.1"],
    },
    install_requires = ['numpy', 'pandas', 'tqdm'],
    tests_require = ['nose'],
    test_suite = 'nose.collector'
)
