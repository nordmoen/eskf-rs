//! # Error State Kalman Filter (ESKF)
//! An [Error State Kalman Filter](https://arxiv.org/abs/1711.02508) is a navigation filter based
//! on regular Kalman filters, more specifically [Extended Kalman
//! Filters](https://en.wikipedia.org/wiki/Extended_Kalman_filter), that model the "error state" of
//! the system instead of modelling the movement of the system.
//!
//! The navigation filter is used to track `position`, `velocity` and `orientation` of an object
//! which is sensing its state through an [Inertial Measurement Unit
//! (IMU)](https://en.wikipedia.org/wiki/Inertial_measurement_unit) and some other means such as
//! GPS, LIDAR or visual odometry.
//!
//! ## Usage
//! ```
//! use eskf;
//! use nalgebra::{Vector3, Point3};
//! use std::time::Duration;
//!
//! // Create a default filter, modelling a perfect IMU without drift
//! let mut filter = eskf::Builder::new().build();
//! // Read measurements from IMU
//! let imu_acceleration = Vector3::new(0.0, 0.0, -9.81);
//! let imu_rotation = Vector3::zeros();
//! // Tell the filter about what we just measured
//! filter.predict(imu_acceleration, imu_rotation, Duration::from_millis(1000));
//! // Check the new state of the filter
//! // filter.position or filter.velocity...
//! // ...
//! // After some time we get an observation of the actual state
//! filter.observe_position(Point3::new(0.0, 0.0, 0.0), eskf::ESKF::symmetric_variance(0.1));
//! // Since we have supplied an observation of the actual state of the filter the states have now
//! // been updated. The uncertainty of the filter is also updated to reflect this new information.
//! ```
use nalgebra::{Matrix3, MatrixMN, MatrixN, Point3, UnitQuaternion, Vector3, U1, U18, U3};
use std::ops::{AddAssign, SubAssign};
use std::time::Duration;

/// Builder for [`ESKF`]
#[derive(Copy, Clone, Default, Debug)]
pub struct Builder {
    var_acc: Vector3<f32>,
    var_rot: Vector3<f32>,
    var_acc_bias: Vector3<f32>,
    var_rot_bias: Vector3<f32>,
    process_covariance: f32,
}

impl Builder {
    /// Create a new `ESKF` builder with which to configure an `ESKF`
    pub fn new() -> Self {
        Builder::default()
    }

    /// Set the acceleration variance of the IMU system being modeled
    ///
    /// The variance should be `m/s²`
    pub fn acceleration_variance(mut self, var: f32) -> Self {
        self.var_acc = Vector3::from_element(var.powi(2));
        self
    }

    /// Set the acceleration variance of the IMU system being modeled
    ///
    /// The variance should be a vector in [`m/s²`, 3]
    pub fn acceleration_variance_from_vec(mut self, var: Vector3<f32>) -> Self {
        self.var_acc = var.map(|e| e.powi(2));
        self
    }

    /// Set the rotation variance of the IMU system being modeled
    ///
    /// The variance should be `rad/s`
    pub fn rotation_variance(mut self, var: f32) -> Self {
        self.var_rot = Vector3::from_element(var.powi(2));
        self
    }

    /// Set the rotation variance of the IMU system being modeled
    ///
    /// The variance should a vector in [`rad/s`, 3]
    pub fn rotation_variance_from_vec(mut self, var: Vector3<f32>) -> Self {
        self.var_rot = var.map(|e| e.powi(2));
        self
    }

    /// Set the acceleration bias of the IMU system being modeled
    ///
    /// The bias should be `m/(s²sqrt(s))`
    pub fn acceleration_bias(mut self, bias: f32) -> Self {
        self.var_acc_bias = Vector3::from_element(bias.powi(2));
        self
    }

    /// Set the acceleration bias of the IMU system being modeled
    ///
    /// The bias should be a vector in [`m/(s²sqrt(s))`, 3]
    pub fn acceleration_bias_from_vec(mut self, bias: Vector3<f32>) -> Self {
        self.var_acc_bias = bias.map(|e| e.powi(2));
        self
    }

    /// Set the rotation bias of the IMU system being modeled
    ///
    /// The bias should be `rad/(s sqrt(s))`
    pub fn rotation_bias(mut self, bias: f32) -> Self {
        self.var_rot_bias = Vector3::from_element(bias.powi(2));
        self
    }

    /// Set the rotation bias of the IMU system being modeled
    ///
    /// The bias should a vector in [`rad/(s sqrt(s))`, 3]
    pub fn rotation_bias_from_vec(mut self, bias: Vector3<f32>) -> Self {
        self.var_rot_bias = bias.map(|e| e.powi(2));
        self
    }

    /// Set the initial covariance for the process matrix
    ///
    /// The covariance value should be a small process value so that the covariance of the filter
    /// quickly converges to the correct value. Too small values could lead to the filter taking a
    /// long time to converge and report a lower covariance than what it should.
    pub fn initial_covariance(mut self, covar: f32) -> Self {
        self.process_covariance = covar;
        self
    }

    /// Convert the builder into a new filter
    pub fn build(self) -> ESKF {
        ESKF {
            position: Point3::origin(),
            velocity: Vector3::zeros(),
            orientation: UnitQuaternion::identity(),
            accel_bias: Vector3::zeros(),
            rot_bias: Vector3::zeros(),
            gravity: Vector3::new(0f32, 0f32, -9.81),
            covariance: MatrixN::<f32, U18>::identity() * self.process_covariance,
            var_acc: self.var_acc,
            var_rot: self.var_rot,
            var_acc_bias: self.var_acc_bias,
            var_rot_bias: self.var_rot_bias,
        }
    }
}

/// Error State Kalman Filter
///
/// The filter works by calling [`predict`](ESKF::predict) and one or more of the `observe_`
/// methods when data is available.
///
/// The [`predict`](ESKF::predict) step updates the internal state of the filter based on measured
/// acceleration and rotation coming from an IMU. This step updates the states in the filter based
/// on kinematic motion while increasing the uncertainty of the filter. When one of the `observe_`
/// methods are called, the filter updates the internal state based on this observation, which
/// exposes the error state to the filter, which we can then use to correct the internal state. The
/// uncertainty of the filter is also updated to reflect the variance of the observation and the
/// updated state.
#[derive(Copy, Clone, Debug)]
pub struct ESKF {
    /// Estimated position in filter
    pub position: Point3<f32>,
    /// Estimated velocity in filter
    pub velocity: Vector3<f32>,
    /// Estimated orientation in filter
    pub orientation: UnitQuaternion<f32>,
    /// Estimated acceleration bias
    pub accel_bias: Vector3<f32>,
    /// Estimated rotation bias
    pub rot_bias: Vector3<f32>,
    /// Estimated gravity vector
    pub gravity: Vector3<f32>,
    /// Covariance of filter state
    pub covariance: MatrixN<f32, U18>,
    /// Acceleration variance
    var_acc: Vector3<f32>,
    /// Rotation variance
    var_rot: Vector3<f32>,
    /// Acceleration variance bias
    var_acc_bias: Vector3<f32>,
    /// Rotation variance bias
    var_rot_bias: Vector3<f32>,
}

impl ESKF {

    /// Create a symmetric variance matrix based on a single variance element
    pub fn symmetric_variance(var: f32) -> Matrix3<f32> {
        Matrix3::from_diagonal_element(var)
    }

    /// Updated the filter, predicting the new state, based on measured acceleration and rotation
    pub fn predict(&mut self, acceleration: Vector3<f32>, rotation: Vector3<f32>, delta: Duration) {
        let delta_t = delta.as_secs_f32();
        let rot_acc_grav = self
            .orientation
            .transform_vector(&(acceleration - self.accel_bias))
            - self.gravity;
        let norm_rot = UnitQuaternion::from_scaled_axis((rotation - self.rot_bias) * delta_t);
        // Update internal state kinematics
        self.position += self.velocity * delta_t + 0.5 * rot_acc_grav * delta_t.powi(2);
        self.velocity += rot_acc_grav * delta_t;
        self.orientation = self.orientation * norm_rot;

        // Propagate uncertainty, since we have not observed any new information about the state of
        // the filter we need to update our estimate of the uncertainty of the filer
        let ident_delta = Matrix3::<f32>::identity() * delta_t;
        let orient_mat = self.orientation.to_rotation_matrix().into_inner();
        let mut error_jacobian = MatrixN::<f32, U18>::identity();
        error_jacobian
            .fixed_slice_mut::<U3, U3>(0, 3)
            .copy_from(&ident_delta);
        error_jacobian
            .fixed_slice_mut::<U3, U3>(3, 6)
            .copy_from(&(-orient_mat * skew(&(acceleration - self.accel_bias)) * delta_t));
        error_jacobian
            .fixed_slice_mut::<U3, U3>(3, 9)
            .copy_from(&(-orient_mat * delta_t));
        error_jacobian
            .fixed_slice_mut::<U3, U3>(3, 15)
            .copy_from(&ident_delta);
        error_jacobian
            .fixed_slice_mut::<U3, U3>(6, 6)
            .copy_from(&(orient_mat.transpose() * norm_rot.to_rotation_matrix()));
        error_jacobian
            .fixed_slice_mut::<U3, U3>(6, 12)
            .copy_from(&-ident_delta);
        self.covariance = error_jacobian * self.covariance * error_jacobian.transpose();
        // Add noise variance
        let mut diagonal = self.covariance.diagonal();
        diagonal
            .fixed_slice_mut::<U3, U1>(3, 0)
            .add_assign(self.var_acc * delta_t.powi(2));
        diagonal
            .fixed_slice_mut::<U3, U1>(6, 0)
            .add_assign(self.var_rot * delta_t.powi(2));
        diagonal
            .fixed_slice_mut::<U3, U1>(9, 0)
            .add_assign(self.var_acc_bias * delta_t);
        diagonal
            .fixed_slice_mut::<U3, U1>(12, 0)
            .add_assign(self.var_rot_bias * delta_t);
        self.covariance.set_diagonal(&diagonal);
    }

    /// Internal update method to update the filter from a single measurement
    fn update3(
        &mut self,
        jacobian: MatrixMN<f32, U3, U18>,
        difference: Vector3<f32>,
        variance: Matrix3<f32>,
    ) {
        // Correct filter based on Kalman gain
        let kalman_gain = self.covariance
            * jacobian.transpose()
            * (jacobian * self.covariance * jacobian.transpose() + variance)
                .try_inverse()
                .expect("Could not invert Kalman gain");
        let error_state = kalman_gain * difference;
        // Update the covariance based on the observed filter state
        if cfg!(feature = "cov-symmetric") {
            self.covariance -= kalman_gain
                * (jacobian * self.covariance * jacobian.transpose() + variance)
                * kalman_gain.transpose();
        } else if cfg!(feature = "cov-joseph") {
            let step1 = MatrixN::<f32, U18>::identity() - kalman_gain * jacobian;
            let step2 = kalman_gain * variance * kalman_gain.transpose();
            self.covariance = step1 * self.covariance * step1.transpose() + step2;
        } else {
            self.covariance =
                (MatrixN::<f32, U18>::identity() - kalman_gain * jacobian) * self.covariance;
        }
        // Inject error state into nominal
        self.position += error_state.fixed_slice::<U3, U1>(0, 0);
        self.velocity += error_state.fixed_slice::<U3, U1>(3, 0);
        self.orientation *=
            UnitQuaternion::from_scaled_axis(error_state.fixed_slice::<U3, U1>(6, 0));
        self.accel_bias += error_state.fixed_slice::<U3, U1>(9, 0);
        self.rot_bias += error_state.fixed_slice::<U3, U1>(12, 0);
        self.gravity += error_state.fixed_slice::<U3, U1>(15, 0);
        // Perform full ESKF reset
        //
        // Since the orientation error is usually relatively small this step can be skipped, but
        // the full formulation can lead to better stability of the filter
        if cfg!(feature = "full-reset") {
            let mut g = MatrixN::<f32, U18>::identity();
            g.fixed_slice_mut::<U3, U3>(6, 6)
                .sub_assign(0.5 * skew(&error_state.fixed_slice::<U3, U1>(6, 0).clone_owned()));
            self.covariance = self.covariance * g * self.covariance.transpose();
        }
    }

    /// Update the filter with an observation of the position
    pub fn observe_position(&mut self, measurement: Point3<f32>, variance: Matrix3<f32>) {
        let mut jacobian = MatrixMN::<f32, U3, U18>::zeros();
        jacobian
            .fixed_slice_mut::<U3, U3>(0, 0)
            .fill_with_identity();
        let diff = measurement - self.position;
        self.update3(jacobian, diff, variance);
    }

    /// Update the filter with an observation of the velocity
    pub fn observe_velocity(&mut self, measurement: Vector3<f32>, variance: Matrix3<f32>) {
        let mut jacobian = MatrixMN::<f32, U3, U18>::zeros();
        jacobian
            .fixed_slice_mut::<U3, U3>(0, 3)
            .fill_with_identity();
        let diff = measurement - self.velocity;
        self.update3(jacobian, diff, variance);
    }

    /// Update the filter with an observation of the orientation
    pub fn observe_orientation(
        &mut self,
        measurement: UnitQuaternion<f32>,
        variance: Matrix3<f32>,
    ) {
        let mut jacobian = MatrixMN::<f32, U3, U18>::zeros();
        jacobian
            .fixed_slice_mut::<U3, U3>(0, 6)
            .fill_with_identity();
        let diff = measurement * self.orientation;
        self.update3(jacobian, diff.scaled_axis(), variance);
    }
}

/// Create the skew-symmetric matrix from a vector
#[rustfmt::skip]
fn skew(v: &Vector3<f32>) -> Matrix3<f32> {
    Matrix3::new(0., -v.z, v.y,
                 v.z, 0., -v.x,
                 -v.y, v.x, 0.)
}

#[cfg(test)]
mod test {
    use super::Builder;
    use approx::assert_relative_eq;
    use nalgebra::{Point3, UnitQuaternion, Vector3};
    use std::f32::consts::FRAC_PI_2;
    use std::time::Duration;

    #[test]
    fn creation() {
        let filter = Builder::new().build();
        assert_relative_eq!(filter.position, Point3::origin());
        assert_relative_eq!(filter.velocity, Vector3::zeros());
    }

    #[test]
    fn linear_motion() {
        let mut filter = Builder::new().build();
        // Some initial motion to move the filter
        filter.predict(
            Vector3::new(1.0, 0.0, -9.81),
            Vector3::zeros(),
            Duration::from_millis(1000),
        );
        assert_relative_eq!(filter.position, Point3::new(0.5, 0.0, 0.0));
        assert_relative_eq!(filter.velocity, Vector3::new(1.0, 0.0, 0.0));
        // There should be no orientation change from the above motion
        assert_relative_eq!(filter.orientation, UnitQuaternion::identity());
        // Acceleration has stopped, but there will still be inertia in the filter
        filter.predict(
            Vector3::new(0.0, 0.0, -9.81),
            Vector3::zeros(),
            Duration::from_millis(500),
        );
        assert_relative_eq!(filter.position, Point3::new(1.0, 0.0, 0.0));
        assert_relative_eq!(filter.velocity, Vector3::new(1.0, 0.0, 0.0));
        assert_relative_eq!(filter.orientation, UnitQuaternion::identity());
        filter.predict(
            Vector3::new(-1.0, -1.0, -9.81),
            Vector3::zeros(),
            Duration::from_millis(1000),
        );
        assert_relative_eq!(filter.position, Point3::new(1.5, -0.5, 0.0));
        assert_relative_eq!(filter.velocity, Vector3::new(0.0, -1.0, 0.0));
        assert_relative_eq!(filter.orientation, UnitQuaternion::identity());
    }

    #[test]
    fn rotational_motion() {
        let mut filter = Builder::new().build();
        // Note that this motion is a free fall rotation
        filter.predict(
            Vector3::zeros(),
            Vector3::new(FRAC_PI_2, 0.0, 0.0),
            Duration::from_millis(1000),
        );
        assert_relative_eq!(
            filter.orientation,
            UnitQuaternion::from_euler_angles(FRAC_PI_2, 0.0, 0.0)
        );
        filter.predict(
            Vector3::zeros(),
            Vector3::new(-FRAC_PI_2, 0.0, 0.0),
            Duration::from_millis(1000),
        );
        assert_relative_eq!(
            filter.orientation,
            UnitQuaternion::from_euler_angles(0.0, 0.0, 0.0)
        );
        // We reset the filter here so that the following equalities are not affected by existing
        // motion in the filter
        let mut filter = Builder::new().build();
        filter.predict(
            Vector3::zeros(),
            Vector3::new(0.0, -FRAC_PI_2, 0.0),
            Duration::from_millis(1000),
        );
        assert_relative_eq!(
            filter.orientation,
            UnitQuaternion::from_euler_angles(0.0, -FRAC_PI_2, 0.0)
        );
    }
}
