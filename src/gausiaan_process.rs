use crate::kernel_defs::Kernel;

use ndarray::{Array1, Array2, ArrayView1};
use ndarray_linalg::Solve;

pub struct GaussianProcess<K: Kernel> {
    pub x_train: Array1<f64>,
    pub y_train: Array1<f64>,
    pub kernel: K,
    pub gram_matrix: Array2<f64>,
}


impl<K: Kernel> GaussianProcess<K> {

    pub fn new(
        x_train: ArrayView1<f64>,
        y_train: ArrayView1<f64>,
        kernel: K) -> GaussianProcess<K> {
        let gram_matrix = compute_gram_matrix(x_train.view(), x_train.view(), &kernel);
        return Self {
            x_train: x_train.to_owned(),
            y_train: y_train.to_owned(),
            kernel: kernel,
            gram_matrix: gram_matrix,
        };
    }

    pub fn optimize_hyperparameters(&mut self) -> () {
        let mut max_likelihood = f64::NEG_INFINITY;
        let mut best_params = self.kernel.get_hyper_params();
        loop {
            let gram_matrix = compute_gram_matrix(self.x_train.view(), self.x_train.view(), &self.kernel);
            let likelihood = self.kernel.compute_likelihood(gram_matrix.view(), self.y_train.view());

            if likelihood > max_likelihood {
                max_likelihood = likelihood;
                best_params = self.kernel.get_hyper_params();
            }
            if !self.kernel.to_next_param() {
                break;
            }
        }
        print!("best hyper params: ");
        for param in best_params.iter() {
            print!("{:.2} ", param.value);
        }
        self.kernel.set_hyper_params(best_params);

    }

    pub fn predict(
        &mut self,
        x_star: ArrayView1<f64>) -> (Array1<f64>, Array1<f64>) {

        let k_star = compute_gram_matrix(self.x_train.view(),x_star.view(), &self.kernel);
        let k_star_t = k_star.t();

        // --- compute mean: transpose(k_star) * inv(gram) * y_train ---
        self.gram_matrix = compute_gram_matrix(self.x_train.view(), self.x_train.view(), &self.kernel);
        let gram_inv_y = self.gram_matrix.solve(&self.y_train).unwrap();
        let mean = k_star_t.dot(&gram_inv_y);

        // --- compute variance: karkel(x_star, x_star) - diag(k_star_t * inv(gram) * k_star) ---
        let x_train_len = self.x_train.len();
        let mut gram_inv = Array2::<f64>::zeros((x_train_len, x_train_len));
        for i in 0..x_train_len {
            let mut unit_vector = Array1::<f64>::zeros(x_train_len);
            unit_vector[i] = 1.0;
            let col = self.gram_matrix.solve(&unit_vector).unwrap();
            gram_inv.column_mut(i).assign(&col);
        }

        let variance_matrix = k_star_t.dot(&gram_inv).dot(&k_star);
        let k_star_star = compute_gram_matrix(x_star.view(), x_star.view(), &self.kernel).diag().to_owned();
        let y_star_variance = k_star_star - variance_matrix.diag().to_owned();

        let sigma = y_star_variance.mapv(f64::sqrt);

        return (mean, sigma);
    }

}

fn compute_gram_matrix<K: Kernel>(
    x1: ArrayView1<f64>,
    x2: ArrayView1<f64>,
    kernel: &K) -> Array2<f64> {

    let mut mat = Array2::<f64>::zeros((x1.len(), x2.len()));
    for (i, x1_elem) in x1.iter().enumerate() {
        for (j, x2_elem) in x2.iter().enumerate() {
            mat[[i, j]] = kernel.compute(*x1_elem, *x2_elem);
        }
    }
    return mat;
}