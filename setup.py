from setuptools import setup, find_packages

setup(
    name="pixelprune",
    version="1.0.0",
    description="PixelPrune: Pixel-Level Adaptive Visual Token Reduction via Predictive Coding",
    author="Nan Wang",
    license="Apache-2.0",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.1.0",
        "transformers>=4.45.0",
        "Pillow>=10.0.0",
        "numpy>=1.24.0",
    ],
)
