use ndarray::{ArrayView1, ArrayView2};
use ndarray_linalg::{Determinant, Solve};

#[derive(Clone)]
pub struct Parameter {
    pub value: f64,
    pub min: f64,
    pub max: f64,
    pub step: f64,
}

impl Parameter {
    pub fn new(min: f64, max: f64, step: f64) -> Parameter {
        return Self {
            value: min,
            min: min,
            max: max,
            step: step,
        };
    }
}

pub trait Kernel {
    fn compute(&self, x1: f64, x2: f64) -> f64;
    fn to_next_param(&mut self) -> bool {true}
    fn compute_likelihood(&self, gram_matrix: ArrayView2<f64>, y: ArrayView1<f64>) -> f64;
    fn get_param(&self) -> Vec<Parameter>;
}

// Radial Basis Function
#[derive(Clone)]
pub struct RBFKernel {
    pub theta1: Parameter,
    pub theta2: Parameter,
}

impl RBFKernel {
    pub fn new(theta1: Parameter, theta2: Parameter) -> RBFKernel {
        return Self {
            theta1: theta1,
            theta2: theta2,
        };
    }
}

impl Kernel for RBFKernel {
    fn compute(&self, x1: f64, x2: f64) -> f64 {
        return self.theta1.value * f64::exp(-(x1 - x2).powi(2) / self.theta2.value);
    }

    fn to_next_param(&mut self) -> bool {
        self.theta1.value += self.theta1.step;
        if self.theta1.value > self.theta1.max {
            self.theta1.value = self.theta1.min;
            self.theta2.value += self.theta2.step;
            if self.theta2.value > self.theta2.max {
                return false;
            }
        }
        return true;
    }

    fn compute_likelihood(&self, gram_matrix: ArrayView2<f64>, y: ArrayView1<f64>) -> f64 {
        // --- compute log(det(gram)) ---
        let gram_det = gram_matrix.det().unwrap();

        // logalityhmic likelihood
        return gram_det.ln() + y.dot(&gram_matrix.solve(&y).unwrap());
    }

    fn get_param(&self) -> Vec<Parameter> {
        return vec![self.theta1.clone(), self.theta2.clone()];
    }

}