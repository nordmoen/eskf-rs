# Error State Kalman Filter (`ESKF`)
[![Continuous integration](https://github.com/nordmoen/eskf-rs/actions/workflows/ci.yml/badge.svg)](https://github.com/nordmoen/eskf-rs/actions/workflows/ci.yml)

This crate implements a navigation filter based on an [Error State Kalman
Filter](./docs/Error_State_Kalman_Filter.pdf).

This crate supports `no_std` environments, but few optimizations to the
mathematics have been attempted to optimize for `no_std`.

## Error State Kalman Filter (ESKF)
An [Error State Kalman Filter](https://arxiv.org/abs/1711.02508) is a navigation
filter based on regular Kalman filters, more specifically [Extended Kalman
Filters](https://en.wikipedia.org/wiki/Extended_Kalman_filter), that model the
"error state" of the system instead of modelling the movement of the system
explicitly.

The navigation filter is used to track `position`, `velocity` and `orientation`
of an object which is sensing its state through an [Inertial Measurement Unit
(IMU)](https://en.wikipedia.org/wiki/Inertial_measurement_unit) and some means
of observing the true state of the filter such as GPS, LIDAR or visual odometry.

## Usage
```rust
use eskf;
use nalgebra::{Vector3, Point3};
use std::time::Duration;

// Create a default filter, modelling a perfect IMU without drift
let mut filter = eskf::Builder::new().build();
// Read measurements from IMU
let imu_acceleration = Vector3::new(0.0, 0.0, -9.81);
let imu_rotation = Vector3::zeros();
// Tell the filter what we just measured
filter.predict(imu_acceleration, imu_rotation, Duration::from_millis(1000));
// Check the new state of the filter
// filter.position or filter.velocity...
// ...
// After some time we get an observation of the actual state
filter.observe_position(
    Point3::new(0.0, 0.0, 0.0),
    eskf::ESKF::variance_from_element(0.1))
        .expect("Filter update failed");
// Since we have supplied an observation of the actual state of the filter the states have now
// been updated. The uncertainty of the filter is also updated to reflect this new information.
```
