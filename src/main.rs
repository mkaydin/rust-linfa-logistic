use std::io::Read;

use csv::ReaderBuilder;
use flate2::read::GzDecoder;
use linfa::{Dataset, prelude::{Records, ToConfusionMatrix}, traits::{Fit, Predict, Transformer}};
use ndarray::prelude::*;
use ndarray_csv::*;
use std::error::Error;
use linfa_logistic::LogisticRegression;
use linfa_preprocessing::linear_scaling::LinearScaler;

pub fn array_from_csv<R: Read>(
    csv: R,
    has_headers: bool,
    seperator: u8,

)-> Result<Array2<f64>, ReadError>{
    let mut reader = ReaderBuilder::new()
        .has_headers(has_headers)
        .delimiter(seperator)
        .from_reader(csv);

    // extract ndarray
    reader.deserialize_array2_dynamic()
}

pub fn array_from_csv_gz<R: Read>(
    gz: R,
    has_headers: bool,
    seperator: u8,
)-> Result<Array2<f64>, ReadError>{
  let file = GzDecoder::new(gz);
  array_from_csv(file, has_headers, seperator)
}

pub fn winequality() -> Dataset<f64, usize, Ix1> {
    let data = include_bytes!("../winequality-red.csv.gz");
    let array = array_from_csv_gz(&data[..],true,b',').unwrap();

    let (data, targets) = (
        array.slice(s![..,0..11]).to_owned(),
        array.column(11).to_owned(),
    );

    let feature_names = vec![
        "fixed acidity",
        "volatile acidity",
        "citric acid",
        "residual sugar",
        "chlorides",
        "free sulfur dioxide",
        "total sulfur dioxide",
        "density",
        "pH",
        "sulphates",
        "alcohol",
    ];

    Dataset::new(data, targets)
        .map_targets(|x| *x as usize)
        .with_feature_names(feature_names)
}

pub fn logistic_classification(){
    let (train, valid) = winequality()
        .map_targets(|x| if *x > 6 {"good"} else {"bad"})
        .split_with_ratio(0.9);
    println!("fit logistic regression with {} training points", train.nsamples());

    let model = LogisticRegression::default()
        .max_iterations(150)
        .fit(&train)
        .unwrap();

    let pred = model.predict(&valid);
    let cm = pred.confusion_matrix(&valid).unwrap();
    println!("{:?}", cm);

    println!("accuracy: {}, MCC {}", cm.accuracy(), cm.mcc());
}

pub fn preprocess_linear_scaling_logistic(){
    let (train, valid) = winequality()
        .map_targets(|x| if *x > 6 {"good"} else {"bad"})
        .split_with_ratio(0.9);
    println!("fit logistic regression with {} training points", train.nsamples());

    let scaler = LinearScaler::standard().fit(&train).unwrap();
    let train_pre = scaler.transform(train);
    let valid_pre = scaler.transform(valid);

    let model_pre = LogisticRegression::default()
        .max_iterations(150)
        .fit(&train_pre)
        .unwrap();
    let pred_pre = model_pre.predict(&valid_pre);
    let cm_pre = pred_pre.confusion_matrix(&valid_pre).unwrap();

    println!("{:?}", cm_pre);
    println!("accuracy: {}, MCC {}", cm_pre.accuracy(), cm_pre.mcc());
}

fn main() -> Result<(), Box<dyn Error>> {
    
    logistic_classification();
    preprocess_linear_scaling_logistic();

    Ok (())
}
