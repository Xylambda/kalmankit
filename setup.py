from setuptools import setup, find_packages
import versioneer


setup(
    name='kalmanfilter',
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    packages=find_packages(exclude=['tests', 'docs', 'examples', 'img']),
    description='General implementation of Kalman Filter algorithm',
    author='Alejandro Pérez',
    install_requires=[
        "numpy",
        "matplotlib"
    ],
)