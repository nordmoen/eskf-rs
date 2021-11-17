//! Example which reads data from the CARLA dataset and plots filter state
use nalgebra::{Point3, Vector3};
use plotly::common::{ErrorData, ErrorType};
use plotly::layout::{GridPattern, Layout, LayoutGrid};
use plotly::{Plot, Scatter};
use serde::Deserialize;
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

#[derive(Default)]
struct PlotLine {
    x: Vec<f32>,
    measured: Vec<f32>,
    error: Vec<f64>,
    truth: Vec<f32>,
}

#[derive(Default)]
struct PlotRow {
    x: PlotLine,
    y: PlotLine,
    z: PlotLine,
}

impl PlotRow {
    fn add(&mut self, i: f32, measured: &Vector3<f32>, truth: &Vector3<f32>, error: &Vector3<f32>) {
        self.x.x.push(i);
        self.x.measured.push(measured.x);
        self.x.error.push(error.x as f64);
        self.x.truth.push(truth.x);

        self.y.x.push(i);
        self.y.measured.push(measured.y);
        self.y.error.push(error.y as f64);
        self.y.truth.push(truth.y);

        self.z.x.push(i);
        self.z.measured.push(measured.z);
        self.z.error.push(error.z as f64);
        self.z.truth.push(truth.z);
    }
}

fn plot_lines(lines: Vec<PlotLine>) {
    let mut plot = Plot::new();
    for (i, line) in lines.iter().enumerate() {
        let x_axis = format!("x{}", i + 1);
        let y_axis = format!("y{}", i + 1);
        let trace = Scatter::new(line.x.clone(), line.measured.clone())
            .x_axis(&x_axis)
            .y_axis(&y_axis)
            .error_y(ErrorData::new(ErrorType::Data).array(line.error.clone()));
        let truth = Scatter::new(line.x.clone(), line.truth.clone())
            .x_axis(&x_axis)
            .y_axis(&y_axis);
        plot.add_trace(trace);
        plot.add_trace(truth);
    }
    let layout = Layout::new().grid(
        LayoutGrid::new()
            .rows(lines.len() / 3)
            .columns(3)
            .pattern(GridPattern::Independent),
    );
    plot.set_layout(layout);
    plot.show();
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dataset_name = std::env::args()
        .nth(1)
        .unwrap_or("tests/carla_dataset1.json".to_owned());
    // Read dataset from JSON
    let file = File::open(dataset_name)?;
    let reader = BufReader::new(file);
    let dataset: Dataset = serde_json::from_reader(reader)?;
    let position_variance =
        eskf::ESKF::variance_from_element(dataset.variance.gnss_position.sqrt());
    // Create lines that we want to plot
    let mut plot_pos = PlotRow::default();
    let mut plot_vel = PlotRow::default();
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
    for (i, m) in dataset.data.iter().skip(1).enumerate() {
        let delta = std::time::Duration::from_secs_f32(m.time - last_time);
        last_time = m.time;

        filter.predict(m.imu.acceleration, m.imu.rotation, delta);
        if let Some(position) = m.gnss.position {
            filter
                .observe_position(position, position_variance)
                .expect("Filter observation failed");
        }
        plot_pos.add(
            i as f32,
            &filter.position.coords,
            &m.ground_truth.position.coords,
            &filter.position_uncertainty(),
        );
        plot_vel.add(
            i as f32,
            &filter.velocity,
            &m.ground_truth.velocity,
            &filter.velocity_uncertainty(),
        );
    }
    plot_lines(vec![
        plot_pos.x, plot_pos.y, plot_pos.z, plot_vel.x, plot_vel.y, plot_vel.z,
    ]);
    Ok(())
}
