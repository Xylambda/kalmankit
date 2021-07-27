<p align="center">
  <img src="img/logo.png" width="700">
</p>

![GitHub tag (latest by date)](https://img.shields.io/github/v/tag/Xylambda/kalmanfilter?label=VERSION&style=for-the-badge)
![GitHub code size in bytes](https://img.shields.io/github/languages/code-size/Xylambda/kalmanfilter?style=for-the-badge)
![GitHub issues](https://img.shields.io/github/issues/Xylambda/kalmanfilter?style=for-the-badge)
![Travis (.org)](https://img.shields.io/travis/xylambda/kalmanfilter?style=for-the-badge)
[![doc](https://img.shields.io/badge/DOCS-documentation-blue.svg?style=for-the-badge)](https://xylambda.github.io/kalmanfilter/)

General multidimensional implementation of the Kalman filter algorithm using 
NumPy. The Kalman filter is an optimal estimation algorithm: if the noises
are gaussian and the dynamic system we are modeling is linear, the Kalman 
filter will find the best possible solution.

The Kalman filter estimates a process by using a form of feedback 
control loop: time update (predict) and measurement update (correct/update).


## Installation
Normal user:
```bash
git clone https://github.com/Xylambda/kalmanfilter.git
pip install kalmanfilter/.
```

alternatively:
```bash
git clone https://github.com/Xylambda/kalmanfilter.git
pip install kalmanfilter/. -r kalmanfilter/requirements.txt
```

Developer:
```bash
git clone https://github.com/Xylambda/kalmanfilter.git
pip install -e kalmanfilter/. -r kalmanfilter/requirements-dev.txt
```

## Tests
To run tests you must install the library as a `developer`.
```bash
cd kalmanfilter/
pytest -v tests/
```

## Usage
The library provides 2 examples of usage:
1. [Moving Average](examples/moving_average.py)
1. [Market Beta estimation](examples/market_beta.py)

## References
* Matlab - [Understanding Kalman Filters](https://www.youtube.com/playlist?list=PLn8PRpmsu08pzi6EMiYnR-076Mh-q3tWr)

* Bilgin's Blog - [Kalman filter for dummies](http://bilgin.esme.org/BitsAndBytes/KalmanFilterforDummies)

* Greg Welch, Gary Bishop - [An Introduction to the Kalman Filter](https://www.cs.unc.edu/~welch/media/pdf/kalman_intro.pdf)

* Simo Särkkä - Bayesian filtering and Smoothing. Cambridge University Press.