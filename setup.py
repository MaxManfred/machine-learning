from setuptools import setup, find_packages

__version__ = "0.0.1"

# INSTALL_REQUIRES = [
# ]

TESTS_REQUIRE = [
    'unittest',
]

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='python-machine-learning',
    version=__version__,
    python_requires='>=3.6.0',
    description='Machine Learning Example gallery',
    author='Massimo Manfredino',
    author_email='massimo.manfredino@gmail.it',
    packages=find_packages(),
    namespace_packages=['ml'],
    install_requires=requirements,
    extras_require={'test': TESTS_REQUIRE},
    include_package_data=True
)
