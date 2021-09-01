import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="act3-rta-have-deepsky",
    version="0.0.1",
    author="",
    author_email="",
    description="",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        # "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6, <3.9',
    install_requires=[
        "tensorflow==2.4",
        "ray[rllib]==1.6",
        "tqdm==4.59.0",
        "jsonlines==2.0.0",
        "matplotlib==3.3.4",
        "pytest==6.2.4",
        "flatten_json==0.1.13",
        "aioredis==1.3.1",
        "pyglet==1.5.19",
    ]
)
