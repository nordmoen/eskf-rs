# Error State Kalman Filter (`ESKF`)
This crate implements a navigation filter based on an [Error State Kalman
Filter](docs/Quaternion Kinematics for the Error State Kalman Filter.pdf).

This crate supports `no_std` environments, but few optimizations to the
mathematics have been attempted to optimize for `no_std`.
