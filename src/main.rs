mod gaussian_process;
mod kernel_defs;
mod plot_utils;

use ndarray::Array1;
use plotters::prelude::*;
use rand::Rng;
use rand_distr::{Distribution, Normal};
use std::f64::consts::PI;

const NUM_TRAIN: i32 = 15;
const X_TRAIN_MIN: f64 = -PI;
const X_TRAIN_MAX: f64 = PI;
const X_STAR_MIN: f64 = -1.25 * PI;
const X_STAR_MAX: f64 = 1.25 * PI;
const NOISE_STD: f64 = 0.1;

const CHART_X_MIN: f64 = X_STAR_MIN;
const CHART_X_MAX: f64 = X_STAR_MAX;
const CHART_Y_MIN: f64 = -1.5;
const CHART_Y_MAX: f64 = 1.5;

fn sin_func(x: f64) -> f64 {
    let amp = 1.0;
    let omega = 1.0;
    let phase = 0.0;
    amp * f64::sin(omega * x + phase)
}

fn generate_train_data() -> (Array1<f64>, Array1<f64>) {
    let mut rng = rand::thread_rng(); // 乱数ジェネレータを初期化
    let normal_dist = Normal::new(0.0, NOISE_STD).unwrap();
    let x_train: Array1<f64> = (0..NUM_TRAIN)
        .map(|_| rng.gen_range(X_TRAIN_MIN..X_TRAIN_MAX))
        .collect();
    let y_train = x_train.map(|&x| sin_func(x) + normal_dist.sample(&mut rng));

    (x_train, y_train)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    /* Generate training data */
    let (x_train, y_train) = generate_train_data();

    /* Data for prediction */
    let x_star = ndarray::Array::linspace(X_STAR_MIN, X_STAR_MAX, 200);

    /* Setup Gaussian Process model */
    let kernel = kernel_defs::RBFKernel::new(
        kernel_defs::Parameter::new(0.05, 2.0, 0.05),
        kernel_defs::Parameter::new(0.1, 5.0, 0.1),
        kernel_defs::Parameter::new(0.05, 0.15, 0.02),
    );
    let mut gp = gaussian_process::GaussianProcess::new(x_train.view(), y_train.view(), kernel);

    /* Optimize hyperparameters */
    gp.optimize_hyperparameters();

    /* Prediction */
    let (mean, sigma) = gp.predict(x_star.view());
    let upper = &mean + &sigma;
    let lower = &mean - &sigma;

    /* Draw result */
    let root = BitMapBackend::new("images/plot.png", (640, 480)).into_drawing_area();
    root.fill(&WHITE)?;
    let mut chart = ChartBuilder::on(&root)
        .caption("GPR", ("sans-serif", 20).into_font())
        .margin(5)
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(CHART_X_MIN..CHART_X_MAX, CHART_Y_MIN..CHART_Y_MAX)?;

    // Draw grid lines and label the axis
    chart.configure_mesh().draw()?;

    // Draw the latent function
    plot_utils::draw_func(&mut chart, sin_func, (X_STAR_MIN, X_STAR_MAX))?;

    // Draw the training data
    plot_utils::draw_points(
        &mut chart,
        x_train.view(),
        y_train.view(),
        MAGENTA,
        5,
        false,
    )?;

    // Draw the mean and variance of the prediction
    plot_utils::draw_line(&mut chart, x_star.view(), mean.view(), GREEN)?;
    plot_utils::draw_line(&mut chart, x_star.view(), upper.view(), CYAN)?;
    plot_utils::draw_line(&mut chart, x_star.view(), lower.view(), CYAN)?;

    // Save the figure
    root.present()?;

    Ok(())
}
