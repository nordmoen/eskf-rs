//! Integration tests which uses data from the [CARLA simulator](https://carla.org/)
use approx::assert_relative_eq;
use eskf;
use nalgebra::{Point3, UnitQuaternion, Vector3};
use serde::Deserialize;
use serde_json;
use std::fs::File;
use std::io::BufReader;

#[derive(Deserialize, Debug)]
struct Variance {
    imu_acceleration: f32,
    imu_rotation: f32,
    gnss_position: f32,
}

#[derive(Deserialize, Debug)]
struct GroundTruth {
    acceleration: Vector3<f32>,
    velocity: Vector3<f32>,
    position: Point3<f32>,
    orientation_euler: Vector3<f32>,
}

#[derive(Deserialize, Debug)]
struct IMU {
    acceleration: Vector3<f32>,
    rotation: Vector3<f32>,
}

#[derive(Deserialize, Debug)]
struct GNSS {
    position: Option<Point3<f32>>,
}

#[derive(Deserialize, Debug)]
struct Measurement {
    time: f32,
    ground_truth: GroundTruth,
    imu: IMU,
    gnss: GNSS,
}

#[derive(Deserialize, Debug)]
struct Dataset {
    variance: Variance,
    data: Vec<Measurement>,
}

#[test]
fn dataset1() {
    // Read dataset from JSON
    let file =
        File::open("tests/carla_dataset1.json").expect("Could not open 'tests/carla_dataset1.json");
    let reader = BufReader::new(file);
    let dataset: Dataset = serde_json::from_reader(reader).expect("Could not parse JSON");
    let position_variance = eskf::ESKF::symmetric_variance(dataset.variance.gnss_position.sqrt());
    // Create filter based on dataset
    let mut filter = eskf::Builder::new()
        .acceleration_variance(dataset.variance.imu_acceleration)
        .rotation_variance(dataset.variance.imu_rotation)
        .initial_covariance(1e-1)
        .build();
    // Insert a first measurement into the filter
    let m = &dataset.data[0];
    // Time delta is based on viewing the dataset and choosing a small value that could fit
    filter.predict(
        m.imu.acceleration,
        m.imu.rotation,
        std::time::Duration::from_millis(5),
    );
    // Iterate measurements and update filter
    let mut last_time = m.time;
    for m in dataset.data.iter().skip(1) {
        let delta = std::time::Duration::from_secs_f32(m.time - last_time);
        last_time = m.time;

        filter.predict(m.imu.acceleration, m.imu.rotation, delta);
        if let Some(position) = m.gnss.position {
            filter.observe_position(position, position_variance);
        }

        let gt_orient = UnitQuaternion::from_euler_angles(
            m.ground_truth.orientation_euler.x,
            m.ground_truth.orientation_euler.y,
            m.ground_truth.orientation_euler.z,
        );
        assert_relative_eq!(
            filter.position,
            m.ground_truth.position,
            epsilon = filter.position_uncertainty().norm()
        );
        assert_relative_eq!(
            filter.velocity,
            m.ground_truth.velocity,
            epsilon = filter.velocity_uncertainty().norm()
        );
        assert_relative_eq!(
            filter.orientation,
            gt_orient,
            epsilon = filter.orientation_uncertainty().norm()
        );
    }
}
