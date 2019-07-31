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

    /// ステップ関数（段階関数）
    /// 入力が０以下の時は０になり、０より大きいときは１になる
    pub fn step(&self) -> DMatrix<f64> {
        self.dot().map(|i| {
            match i > 0. {
                true => 1.,
                false => 0.
            }
        })
    }

    /// シグモイド関数
    /// 入力した値が大きければ大きいほど１に近づき、小さければ小さいほど０に近づく
    pub fn sigmoid(&self) -> DMatrix<f64> {
        self.dot().map(|i| {1. / (1. + (-i).exp())})
    }

    /// ReLU関数
    /// 入力が０以下の時は０になり、１より大きいときは入力をそのまま出力する
    pub fn relu(&self) -> DMatrix<f64> {
        self.dot().map(|i| i.max(0.))
    }

    /// 恒等関数
    /// 常に入力した値と同じ値をそのまま返す
    /// 出力層で使われる
    pub fn identify(&self) -> DMatrix<f64> {
        self.dot()
    }

    /// ソフトマックス関数
    pub fn softmax(&self) -> DMatrix<f64> {
        let _dot = self.dot();
        let _max = _dot.iter().fold(0.0 / 0.0, |acc, i| i.max(acc));
        let _sum = _dot.iter().fold(0., |acc, i| (i - _max).exp() + acc);
        _dot.map(|i| {(i - _max).exp() / _sum})
    }
}