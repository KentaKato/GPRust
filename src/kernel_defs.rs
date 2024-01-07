pub trait Kernel {
    fn compute(&self, x1: f64, x2: f64) -> f64;
}

// Radial Basis Function
pub struct RBFKernel {
    pub theta1: f64,
    pub theta2: f64,
}

impl Kernel for RBFKernel {
    fn compute(&self, x1: f64, x2: f64) -> f64 {
        return self.theta1 * f64::exp(-(x1 - x2).powi(2) / self.theta2);
    }
}