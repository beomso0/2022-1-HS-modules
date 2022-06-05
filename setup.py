from setuptools import setup, find_packages

with open('requirements.txt') as f:
  requirements = f.read().splitlines()

with open("README.md", "r") as fh:
  long_description = fh.read()

setuptools.setup(
  name="HS_modules", 
  version="0.8.9",
  author="GH",
  author_email="univ3352@gmail.com",
  description="Recommend modules for HS",
  long_description=long_description,
  long_description_content_type="text/markdown",
  url="https://github.com/beomso0/2022-1-HS-modules",
  packages=setuptools.find_packages(),
  classifiers=[
    "Programming Language :: Python :: 3",
  ],
  python_requires='>=3.7',
  # install_requires=requirements,
)