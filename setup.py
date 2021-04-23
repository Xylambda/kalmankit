from setuptools import setup, find_packages
import versioneer


setup(
    name='kalmanfilter',
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    description='General implementation of Kalman Filter algorithm',
    author='Alejandro Pérez',
    install_requires=[
        "numpy",
        "jupyter",
        "matplotlib"
    ],
)