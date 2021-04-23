# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/).

## [1.3.3] - 2021-04-23

### Fixed

- Fix naming in setup.

### Changed
- Update requirements to fix vulnerabilities.


## [1.3.0] - 2021-03-09

### Changed

- Updated `README`.
- Update matmul operations to use @ operator.


## [1.2.0] - 2021-03-08

### Changed

- Updated `quickstart`.
- Updated `README`.

### Fixed

- Fixed naming in tests.


## [1.1.1] - 2021-01-13

### Changed

- Updated `example`.
- Updated `README` with latest changes.

### Fixed

- Fixed docs renderization problems.


## [1.1.0] - 2021-01-13

### Added

- Added `I` attribute (Identity matrix).
- Added `CI` with TravisCI.
- Added formulas in doc for KalmanFilter class.

### Changed

- Updated `predict` method to receive the vector-control input value at time k.
- Updated `test_kalman.py`.
- Updated `README` with latest changes.

### Fixed

- Fixed vector-control input (U). Now it depends on k like Z.
- Fixed minor typos in docs.


## [1.0.1] - 2021-01-10

### Added

- Added `state_size` attribute.

### Changed

- Updated `predict` method. Now it does not return anything.
- Updated `test_kalman.py` to a more friendly syntax.
- Updated problematic dependencies.

### Fixed

- Fixed minor typos in README.


## [1.0.0] - 2021-01-09

### Added

- Added multidimensional computations.
- Added log and badges from [shield.io](https://shields.io/)
- Added `control-input` matrix and vector to computations.

### Changed

- Updated README.
- Updated module design that allows for future features.
- Updated tests for kalman filter.
- Updated kalman_filter. Now it is a class and supports multivariate computations.

### Fixed

- Fixed minor typos in README.
- Fixed `update` formula in README.
