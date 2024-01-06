mod gausiaan_process;

use rand::Rng;
use rand_distr::{Normal, Distribution};
use ndarray::{Array1, ArrayView1};
use std::f64::consts::PI;
use plotters::{prelude::*, coord::types::RangedCoordf64};

fn sin_func(x: f64) -> f64 {
    let amp = 1.0;
    let omega = 1.0;
    let phase = 0.0;
    return amp * f64::sin(omega * x + phase);
}

fn draw_func(
    chart: &mut ChartContext<BitMapBackend, Cartesian2d<RangedCoordf64, RangedCoordf64>>,
    func: fn(f64) -> f64,
    bounds: (f64, f64)) -> Result<(), Box<dyn std::error::Error>> {

    let step = 0.1;
    let range = (0..=((bounds.1 - bounds.0) / step).round() as usize)
        .map(|i| bounds.0 + i as f64 * step)
        .filter(|&x| x <= bounds.1);
    chart.draw_series(LineSeries::new(
        range.map(|x| (x, func(x))),
        &BLUE
    ))?;

    return Ok(());
}

fn draw_points(
    chart: &mut ChartContext<BitMapBackend, Cartesian2d<RangedCoordf64, RangedCoordf64>>,
    x_array: ArrayView1<f64>,
    y_array: ArrayView1<f64>,
    color: RGBColor,
    point_size: i32,
    enable_line: bool) -> Result<(), Box<dyn std::error::Error>> {

    for (x, y) in x_array.iter().zip(y_array.iter()) {
        chart.draw_series(PointSeries::of_element(
            vec![(*x, *y)], // ベクトル (x, y) を作成
            point_size, // 点のサイズ
            ShapeStyle::from(&color).filled(), // 点のスタイル
            &|coord, size, style| { // 無名関数（クロージャ）を作成。&で参照キャプチャを指定。
                EmptyElement::at(coord) // 点の座標を指定（空の要素を配置）
                    + Circle::new((0, 0), size, style) // 円を描画
            },
        ))?;
    }

    if enable_line {
        chart.draw_series(LineSeries::new(
            x_array.iter().zip(y_array.iter()).map(|(&x, &y)| (x, y)),
            &color,
        ))?;
    }
    return Ok(());
}

fn draw_line(
    chart: &mut ChartContext<BitMapBackend, Cartesian2d<RangedCoordf64, RangedCoordf64>>,
    x_array: ArrayView1<f64>,
    y_array: ArrayView1<f64>,
    color: RGBColor) -> Result<(), Box<dyn std::error::Error>> {

    chart.draw_series(LineSeries::new(
        x_array.iter().zip(y_array.iter()).map(|(&x, &y)| (x, y)),
        &color
    ))?;
    return Ok(());
}

// Radial Basis Function
fn rbf_kernel(x1: f64, x2: f64) -> f64 {
    let theta1 = 0.1;
    let theta2 = 0.1;
    return theta1 * f64::exp(-(x1 - x2).powi(2) / theta2);
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

    let gp = gausiaan_process::GaussianProcess::new(x_train.view(), y_train.view(), rbf_kernel);
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

    draw_points(&mut chart, x_train.view(), y_train.view(), MAGENTA, 5, false)?;
    draw_line(&mut chart, x_star.view(), mean.view(), GREEN)?;
    draw_line(&mut chart, x_star.view(), upper.view(), CYAN)?;
    draw_line(&mut chart, x_star.view(), lower.view(), CYAN)?;

    // sin関数を描画
    draw_func(&mut chart, sin_func, (-PI, PI))?;

    // 描画が終わったら、画像ファイルを保存
    root.present()?;

    return Ok(());

}
