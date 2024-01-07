mod gausiaan_process;
mod kernel_defs;
mod plot_utils;

use rand::Rng;
use rand_distr::{Normal, Distribution};
use ndarray::Array1;
use std::f64::consts::PI;
use plotters::prelude::*;

fn sin_func(x: f64) -> f64 {
    let amp = 1.0;
    let omega = 1.0;
    let phase = 0.0;
    return amp * f64::sin(omega * x + phase);
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut rng = rand::thread_rng(); // 乱数ジェネレータを初期化

    const NUM_TRAIN: i32 = 15;
    let normal_dist = Normal::new(0.0, 0.1).unwrap();
    let x_train: Array1::<f64> = (0..NUM_TRAIN)
        .map(|_| {
            return rng.gen_range(-PI..PI);
        }).collect();
    let y_train = x_train.map(|&x| sin_func(x) + normal_dist.sample(&mut rng));

    let x_star = ndarray::Array::linspace(-PI, PI, 100);

    let kernel = kernel_defs::RBFKernel {
        theta1: 0.1,
        theta2: 0.1,
    };
    let gp = gausiaan_process::GaussianProcess::new(
        x_train.view(),
        y_train.view(),
        kernel);
    let (mean, sigma) = gp.predict(x_star.view());
    let upper = &mean + &sigma;
    let lower = &mean - &sigma;


    // グラフを描画する画像ファイルを作成
    let root = BitMapBackend::new("images/plot.png", (640, 480)).into_drawing_area();

    // グラフの背景を白に設定
    root.fill(&WHITE)?;

    // グラフの描画領域を設定
    let mut chart = ChartBuilder::on(&root)
        .caption("GPR", ("sans-serif", 20).into_font())
        .margin(5)
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(-PI..PI, -1.0..1.0)?;

    // XY軸、グリッド線を描画
    chart.configure_mesh().draw()?;

    plot_utils::draw_points(&mut chart, x_train.view(), y_train.view(), MAGENTA, 5, false)?;
    plot_utils::draw_line(&mut chart, x_star.view(), mean.view(), GREEN)?;
    plot_utils::draw_line(&mut chart, x_star.view(), upper.view(), CYAN)?;
    plot_utils::draw_line(&mut chart, x_star.view(), lower.view(), CYAN)?;

    // sin関数を描画
    plot_utils::draw_func(&mut chart, sin_func, (-PI, PI))?;

    // 描画が終わったら、画像ファイルを保存
    root.present()?;

    return Ok(());

}
