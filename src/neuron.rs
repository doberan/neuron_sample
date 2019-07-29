extern crate nalgebra;

use nalgebra::core::{DMatrix};

// ニューロン構造体
pub struct Neuron<'a> {
    bias: &'a DMatrix<f64>,
    data: &'a DMatrix<f64>,
    weight: &'a DMatrix<f64>,
}

// ニューロン実装

impl<'a> Neuron<'a> {
    pub fn new(bias: &'a DMatrix<f64>, data: &'a DMatrix<f64>, weight: &'a DMatrix<f64>) -> Self {
        Neuron { bias: bias, data: data, weight: weight }
    }

    fn dot(&self) -> DMatrix<f64> {
        self.data * self.weight + self.bias
    }

    //
    pub fn step(&self) -> DMatrix<f64> {
        self.dot().map(|i| {
            match i > 0. {
                true => 1.,
                false => 0.
            }
        })
    }

    // シグモイド関数
    pub fn sigmoid(&self) -> DMatrix<f64> {
        self.dot().map(|i| {1. / (1. + (-i).exp())})
    }

    // ReLU関数
    pub fn relu(&self) -> DMatrix<f64> {
        self.dot().map(|i| i.max(0.))
    }

    // 恒等関数
    pub fn identify(&self) -> DMatrix<f64> {
        self.dot()
    }

    pub fn softmax(&self) -> DMatrix<f64> {
        let _dot = self.dot();
        let _max = _dot.iter().fold(0.0 / 0.0, |acc, i| i.max(acc));
        let _sum = _dot.iter().fold(0., |acc, i| (i - _max).exp() + acc);
        _dot.map(|i| {(i - _max).exp() / _sum})
    }
}