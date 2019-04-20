import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="Parallel.GAMIT-pmatheny",
    version="0.0.1",
    author="Peter Matheny",
    author_email="peter.matheny@gmail.com",
    description="Wrapper for PPP and GAMIT",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/demiangomez/Parallel.GAMIT",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=['PyGreSQL',
                      'numpy',
                      'matplotlib',
                      'scipy',
                      'scandir',
                      'tqdm',
                      'dispy',
                      'dirsync',
                      'psutil',
                      'hdf5storage',
                      'libcomcat',
                      'pysftp',
                      'simplekml',
                      'magic',
                      'sklearn',
                      'seaborn']
)
