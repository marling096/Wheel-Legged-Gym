from setuptools import find_packages
from distutils.core import setup

setup(
    name="wheel_legged_gym",
    version="1.0.0",
    author="Hongxi Wang",
    license="BSD-3-Clause",
    packages=find_packages(),
    author_email="wanghongxi2001@outlook.com",
    description="Isaac Gym environments for Wheel Legged Robots",
    install_requires=[
        "isaacgym",
        "matplotlib",
        "scipy>=1.5",
        "tensorboard",
        # TensorBoard 2.14 (last line for Python 3.8) requires protobuf 4.x API
        "protobuf>=4.21.6,<5",
        "setuptools==59.5.0",
        "numpy>=1.16.4",
        "numpy<1.20.0",
        "GitPython",
        "onnx",
    ],
)
