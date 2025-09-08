from setuptools import setup, find_packages

setup(
    name="allegro-visualization",
    version="0.1.0",
    description="Allegro hand visualization tools",
    packages=find_packages(),
    install_requires=[
        "torch",
        "numpy",
        "trimesh",
        "transforms3d",
        "pytorch-kinematics",
    ],
    python_requires=">=3.7",
)
