<p align="center">
  <img src="img/logo.png" width="700">
</p>

![GitHub tag (latest by date)](https://img.shields.io/github/v/tag/Xylambda/kalmankit?label=VERSION&style=badge)
![GitHub code size in bytes](https://img.shields.io/github/languages/code-size/Xylambda/kalmankit?style=badge)
![GitHub issues](https://img.shields.io/github/issues/Xylambda/kalmankit?style=badge)
![workflow](https://github.com/Xylambda/kalmankit/actions/workflows/cicd.yaml/badge.svg)
[![doc](https://img.shields.io/badge/DOCS-documentation-blue.svg?style=badge)](https://xylambda.github.io/kalmankit/)

The Kalman filter is an optimal estimation algorithm: it estimates the true 
state of a signal given that this signal is noisy and/or incomplete. This 
package provides a multidimensional implementation of:
* **Standard Kalman Filter**: if the noises are drawn from a gaussian 
distribution and the underlying system is governed by linear equations, the 
filter will output the best possible estimate of the signal's true state.

* **Extended Kalman Filter**: can deal with nonlinear systems, but it does not
guarantee the optimal estimate. It works by linearizing the function locally
using the Jacobian matrix.


## Installation
**Normal user**
```bash
pip install kalmankit
```

**Developer**
```bash
git clone https://github.com/Xylambda/kalmankit.git
pip install -e kalmankit/. -r kalmankit/requirements-dev.txt
```

## Tests
To run tests you must install the library as a `developer`.
```bash
cd kalmankit/
pytest -v tests/
```

## Usage
The library provides 3 examples of usage:
1. [Moving Average](examples/moving_average.py)
2. [Market Beta estimation](examples/market_beta.py)
3. [Extended Kalman Filter](examples/extended.py)

A `requirements-example.txt` is provided to install the needed dependencies to
run the examples.

## References
* Matlab - [Understanding Kalman Filters](https://www.youtube.com/playlist?list=PLn8PRpmsu08pzi6EMiYnR-076Mh-q3tWr)

* Bilgin's Blog - [Kalman filter for dummies](http://bilgin.esme.org/BitsAndBytes/KalmanFilterforDummies)

* Greg Welch, Gary Bishop - [An Introduction to the Kalman Filter](https://www.cs.unc.edu/~welch/media/pdf/kalman_intro.pdf)

* Simo Särkkä - Bayesian filtering and Smoothing. Cambridge University Press.


## Cite
If you've used this library for your projects please cite it:

```latex
@misc{alejandro2021kalmankit,
  title={kalmankit - Multidimensional implementation of Kalman Filter algorithms},
  author={Alejandro Pérez-Sanjuán},
  year={2021},
  howpublished={\url{https://github.com/Xylambda/kalmankit}},
}
```