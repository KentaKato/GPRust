use ndarray::ArrayView1;
use plotters::{coord::types::RangedCoordf64, prelude::*};

pub fn draw_func(
    chart: &mut ChartContext<BitMapBackend, Cartesian2d<RangedCoordf64, RangedCoordf64>>,
    func: fn(f64) -> f64,
    bounds: (f64, f64),
) -> Result<(), Box<dyn std::error::Error>> {
    let step = 0.1;
    let range = (0..=((bounds.1 - bounds.0) / step).round() as usize)
        .map(|i| bounds.0 + i as f64 * step)
        .filter(|&x| x <= bounds.1);
    chart.draw_series(LineSeries::new(range.map(|x| (x, func(x))), &BLUE))?;

    Ok(())
}

pub fn draw_points(
    chart: &mut ChartContext<BitMapBackend, Cartesian2d<RangedCoordf64, RangedCoordf64>>,
    x_array: ArrayView1<f64>,
    y_array: ArrayView1<f64>,
    color: RGBColor,
    point_size: i32,
    enable_line: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    for (x, y) in x_array.iter().zip(y_array.iter()) {
        chart.draw_series(PointSeries::of_element(
            vec![(*x, *y)],                    // ベクトル (x, y) を作成
            point_size,                        // 点のサイズ
            ShapeStyle::from(&color).filled(), // 点のスタイル
            &|coord, size, style| {
                // 無名関数（クロージャ）を作成。&で参照キャプチャを指定。
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
    Ok(())
}

pub fn draw_line(
    chart: &mut ChartContext<BitMapBackend, Cartesian2d<RangedCoordf64, RangedCoordf64>>,
    x_array: ArrayView1<f64>,
    y_array: ArrayView1<f64>,
    color: RGBColor,
) -> Result<(), Box<dyn std::error::Error>> {
    chart.draw_series(LineSeries::new(
        x_array.iter().zip(y_array.iter()).map(|(&x, &y)| (x, y)),
        &color,
    ))?;
    Ok(())
}
