use rand::Rng;
use rand_distr::{Normal, Distribution};
use ndarray;
use ndarray::{Array1, Array2, ArrayView1};
use ndarray_linalg::solve::Inverse;
use std::f64::consts::PI;
use plotters::{prelude::*, coord::types::RangedCoordf64};

fn sin_func(x: f64) -> f64 {
    let amp = 1.0;
    let omega = 1.0;
    let phase = 0.0;
    return amp * f64::sin(omega * x + phase);
}

fn draw_latent_func(
    chart: &mut ChartContext<BitMapBackend, Cartesian2d<RangedCoordf64, RangedCoordf64>>,
    latent_f: fn(f64) -> f64,
    bounds: (f64, f64)) -> Result<(), Box<dyn std::error::Error>> {

    let step = 0.1;
    let range = (0..=((bounds.1 - bounds.0) / step).round() as usize)
        .map(|i| bounds.0 + i as f64 * step)
        .filter(|&x| x <= bounds.1);
    chart.draw_series(LineSeries::new(
        range.map(|x| (x, latent_f(x))),
        &BLUE
    ))?;

    return Ok(());
}

// Radial Basis Function
fn rbf_kernel(x1: f64, x2: f64) -> f64 {
    let theta1 = 1.0;
    let theta2 = 1.0;
    return theta1 * f64::exp(-(x1 - x2).powi(2) / theta2);
}

fn compute_gram_matrix(
    x: &Array1<f64>,
    kernel: fn(f64, f64) -> f64) -> Array2<f64> {

    let x_len = x.len();
    let mut mat = Array2::<f64>::zeros((x_len, x_len));
    for (i, x1) in x.iter().enumerate() {
        for (j, x2) in x.iter().enumerate() {
            mat[[i, j]] = kernel(*x1, *x2);
        }
    }
    return mat;
}

fn compute_k_star(
    x_train: ArrayView1<f64>,
    x_star: ArrayView1<f64>,
    kernel: fn(f64, f64) -> f64) -> Array2<f64> {

    let mut k_star = Array2::<f64>::zeros((x_train.len(), x_star.len()));
    for (i, x_train_) in x_train.iter().enumerate() {
        for (j, x_star_) in x_star.iter().enumerate() {
            k_star[[i, j]] = kernel(*x_train_, *x_star_);
        }
    }
    return k_star;
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut rng = rand::thread_rng(); // 乱数ジェネレータを初期化

    let num_train = 10;
    let normal_dist = Normal::new(0.0, 0.1).unwrap();
    let x_train: Array1::<f64> = (0..num_train)
        .map(|_| {
            return rng.gen_range(-PI..PI);
        }).collect();
    let y_train = x_train.map(|&x| sin_func(x) + normal_dist.sample(&mut rng));

    let x_star = ndarray::array![-1.0, 0.0, 1.5, 2.1];
    let gram = compute_gram_matrix(&x_train, rbf_kernel);
    let k_star = compute_k_star(x_train.view(), x_star.view(), rbf_kernel);

    // --- compute transpose(k_star) * inv(gram) * y_train ---
    let gram_inv = Inverse::inv(&gram);
    let k_star_t = k_star.t();
    let result = k_star_t.dot(&gram_inv).dot(&y_train);

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

    for (x, y) in x_train.iter().zip(y_train.iter()) {
        chart.draw_series(PointSeries::of_element(
            vec![(*x, *y)], // ベクトル (x, y) を作成
            5, // 点のサイズ
            ShapeStyle::from(&RED).filled(), // 点のスタイル
            &|coord, size, style| { // 無名関数（クロージャ）を作成。&で参照キャプチャを指定。
                EmptyElement::at(coord) // 点の座標を指定（空の要素を配置）
                    + Circle::new((0, 0), size, style) // 円を描画
                    + Text::new(format!("{:.2}", coord.1), (10, 0), ("sans-serif", 15).into_font()) // y の値を表示
            },
        ))?;
    }

    // sin関数を描画
    draw_latent_func(&mut chart, sin_func, (-PI, PI))?;

    // 描画が終わったら、画像ファイルを保存
    root.present()?;

    return Ok(());

}
