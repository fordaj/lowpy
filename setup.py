import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="lowpy", # Replace with your own username
    version="0.0.2",
    author="Andrew Ford",
    author_email="author@example.com",
    description="High level GPU simulations of low level device characteristics in ML algorithms",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/fordaj/lowpy",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)