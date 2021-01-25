import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="lowpy",
    version="0.1.0",
    author="Andrew Ford",
    author_email="author@example.com",
    description="High level GPU simulations of low level device characteristics in ML algorithms",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/fordaj/lowpy",
    packages=setuptools.find_packages(),
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)