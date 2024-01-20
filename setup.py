import setuptools

# with open("README.md", "r") as fh:
#     long_description = fh.read()

setuptools.setup(
    name="hand_detection",
    version="0.1.0",
    # author="Pouyan Firouzabadi",
    # author_email="sppf75@gmail.com",
    # long_description=long_description,
    # long_description_content_type="text/markdown",
    # url="https://github.com/Pouyan97/Hand_Detection",
    packages=setuptools.find_packages(include=['hand_detection*']),
)