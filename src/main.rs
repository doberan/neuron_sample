extern crate nalgebra;

use nalgebra::core::{DMatrix};
use neuron_sample::neuron;

// 1層ニューラルネットワーク_
fn single_neural_nw() {
    // 第一層入力データ.
    let b1 = DMatrix::<f64>::from_iterator(1, 3, [0., 0., 0.].iter().cloned());
    let x1 = DMatrix::<f64>::from_iterator(1, 2, [1., 2.].iter().cloned());
    let w1 = DMatrix::<f64>::from_iterator(2, 3, [1., 2., 3., 4., 5., 6.].iter().cloned());
    let n1 = neuron::Neuron::new(&b1, &x1, &w1);

    // 出力層.
    println!("identify:\n{}", n1.identify());
}

// 多層ニューラルネットワーク.
fn multi_neural_nw() {
    let b1 = DMatrix::<f64>::from_iterator(1, 3, [0.1, 0.2, 0.3].iter().cloned());
    let x1 = DMatrix::<f64>::from_iterator(1, 2, [1., 0.5].iter().cloned());
    let w1 = DMatrix::<f64>::from_iterator(2, 3, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6].iter().cloned());
    let n1 = neuron::Neuron::new(&b1, &x1, &w1);

    let b2 = DMatrix::<f64>::from_iterator(1, 2, [0.1, 0.2].iter().cloned());
    let w2 = DMatrix::<f64>::from_iterator(3, 2, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6].iter().cloned());
    let x2 = n1.sigmoid();
    let n2 = neuron::Neuron::new(&b2, &x2, &w2);

    let b3 = DMatrix::<f64>::from_iterator(1, 2, [0.1, 0.2].iter().cloned());
    let w3 = DMatrix::<f64>::from_iterator(2, 2, [0.1, 0.2, 0.3, 0.4].iter().cloned());
    let x3 = n2.sigmoid();
    let n3 = neuron::Neuron::new(&b3, &x3, &w3);

    println!("identify:\n{}", n3.identify());
}

// main関数.
fn main() {
    // 一層ニューラルネットワーク.
    single_neural_nw();

    // 多層ニューラルネットワーク.
    multi_neural_nw();
}