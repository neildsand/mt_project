# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.6.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# +
import setuptools

with open("README.md", "r") as f:
    long_description = f.read()

setuptools.setup(
    name='software_module',
    version='0.0.1',
    author='Dillon Holder',
    author_email='lholder@caltech.edu',
    description='Package for code used in microtubule catastrophe analysis.',
    long_description=long_description,
    long_description_content_type='ext/markdown',
    packages=setuptools.find_packages(),
    install_requires=["numpy","pandas", "bokeh>=1.4.0"],
    classifiers=(
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ),
)
# -


