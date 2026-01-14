use arrow::array::{Array, Float64Array};
use burn::backend::wgpu::{Wgpu, WgpuDevice};
use burn::nn::pool::AvgPool1dConfig;
use burn::nn::PaddingConfig1d;
use burn::tensor::module::conv1d;
use burn::tensor::ops::ConvOptions;
use burn::tensor::{Tensor, TensorData};
use std::collections::HashMap;
use std::sync::Arc;

type B = Wgpu;

pub struct GpuIndicatorHelper {
    device: WgpuDevice,
    close: Tensor<B, 3>,
    _high: Tensor<B, 3>,
    _low: Tensor<B, 3>,
    close_sq: Tensor<B, 3>,
    length: usize,
}

impl GpuIndicatorHelper {
    pub fn new(close: &[f64], high: &[f64], low: &[f64]) -> Self {
        let device = WgpuDevice::default();
        let length = close.len();
        let shape = [1, 1, length];

        let to_tensor = |data: &[f64]| -> Tensor<B, 3> {
            let flat: Vec<f32> = data.iter().map(|&x| x as f32).collect();
            let tensor_data = TensorData::new(flat, shape);
            Tensor::from_data(tensor_data, &device)
        };

        let t_close = to_tensor(close);
        let t_high = to_tensor(high);
        let t_low = to_tensor(low);
        let t_close_sq = t_close.clone().powf_scalar(2.0);

        Self {
            device,
            close: t_close,
            _high: t_high,
            _low: t_low,
            close_sq: t_close_sq,
            length,
        }
    }

    fn create_padded_array(data: &[f32], length: usize, period: usize) -> Arc<dyn Array> {
        let mut result = Vec::with_capacity(length);
        for _ in 0..(period - 1) {
            result.push(None);
        }
        for &val in data {
            result.push(Some(val as f64));
        }
        while result.len() < length {
            result.push(None);
        }
        result.truncate(length);

        Arc::new(Float64Array::from(result))
    }

    fn tensors_to_map(
        &self,
        tensors: Vec<Tensor<B, 1>>,
        period_list: &[usize],
    ) -> HashMap<usize, Arc<dyn Array>> {
        let mut results = HashMap::new();
        for (i, tensor) in tensors.into_iter().enumerate() {
            let p = period_list[i];
            let data = tensor.into_data().to_vec().unwrap_or_default();
            results.insert(p, Self::create_padded_array(&data, self.length, p));
        }
        results
    }

    pub fn compute_all_smas(
        &self,
        periods: std::ops::RangeInclusive<usize>,
    ) -> HashMap<usize, Arc<dyn Array>> {
        let mut tensors: Vec<Tensor<B, 1>> = Vec::new();
        let period_list: Vec<usize> = periods.collect();

        for &period in &period_list {
            let config = AvgPool1dConfig::new(period)
                .with_stride(1)
                .with_padding(PaddingConfig1d::Valid);
            let pool = config.init();
            let res = pool.forward(self.close.clone());
            tensors.push(res.flatten(0, 2));
        }
        self.tensors_to_map(tensors, &period_list)
    }

    pub fn compute_all_wmas(
        &self,
        periods: std::ops::RangeInclusive<usize>,
    ) -> HashMap<usize, Arc<dyn Array>> {
        let mut tensors: Vec<Tensor<B, 1>> = Vec::new();
        let period_list: Vec<usize> = periods.collect();

        for &period in &period_list {
            let weight_sum = (period * (period + 1)) as f32 / 2.0;
            let weights: Vec<f32> = (1..=period).map(|i| i as f32 / weight_sum).collect();
            let weight_tensor =
                Tensor::from_data(TensorData::new(weights, [1, 1, period]), &self.device);
            let options = ConvOptions {
                stride: [1],
                padding: [0],
                dilation: [1],
                groups: 1,
            };
            let res = conv1d(self.close.clone(), weight_tensor, None, options);
            tensors.push(res.flatten(0, 2));
        }
        self.tensors_to_map(tensors, &period_list)
    }

    pub fn compute_all_std_devs(
        &self,
        periods: std::ops::RangeInclusive<usize>,
    ) -> (
        HashMap<usize, Arc<dyn Array>>,
        HashMap<usize, Arc<dyn Array>>,
    ) {
        let mut std_tensors: Vec<Tensor<B, 1>> = Vec::new();
        let mut var_tensors: Vec<Tensor<B, 1>> = Vec::new();
        let period_list: Vec<usize> = periods.collect();

        for &period in &period_list {
            let config = AvgPool1dConfig::new(period)
                .with_stride(1)
                .with_padding(PaddingConfig1d::Valid);
            let pool = config.init();
            let sma_x = pool.forward(self.close.clone());
            let sma_x2 = pool.forward(self.close_sq.clone());
            let variance = sma_x2.sub(sma_x.powf_scalar(2.0));
            let std_dev = variance.clone().clamp_min(0.0).sqrt();

            std_tensors.push(std_dev.flatten(0, 2));
            var_tensors.push(variance.flatten(0, 2));
        }

        (
            self.tensors_to_map(std_tensors, &period_list),
            self.tensors_to_map(var_tensors, &period_list),
        )
    }

    pub fn compute_all_envelopes(
        &self,
        periods: std::ops::RangeInclusive<usize>,
    ) -> HashMap<usize, (Arc<dyn Array>, Arc<dyn Array>, Arc<dyn Array>)> {
        let mut tensors_u: Vec<Tensor<B, 1>> = Vec::new();
        let mut tensors_l: Vec<Tensor<B, 1>> = Vec::new();
        let mut tensors_m: Vec<Tensor<B, 1>> = Vec::new();
        let period_list: Vec<usize> = periods.collect();

        for &period in &period_list {
            let config = AvgPool1dConfig::new(period)
                .with_stride(1)
                .with_padding(PaddingConfig1d::Valid);
            let pool = config.init();
            let sma = pool.forward(self.close.clone());
            let upper = sma.clone().mul_scalar(1.025);
            let lower = sma.clone().mul_scalar(0.975);

            tensors_u.push(upper.flatten(0, 2));
            tensors_l.push(lower.flatten(0, 2));
            tensors_m.push(sma.flatten(0, 2));
        }

        let map_u = self.tensors_to_map(tensors_u, &period_list);
        let map_l = self.tensors_to_map(tensors_l, &period_list);
        let map_m = self.tensors_to_map(tensors_m, &period_list);
        let mut results = HashMap::new();
        for &p in &period_list {
            let empty = Arc::new(Float64Array::from(Vec::<Option<f64>>::new())) as Arc<dyn Array>;
            let u = map_u.get(&p).cloned().unwrap_or(empty.clone());
            let l = map_l.get(&p).cloned().unwrap_or(empty.clone());
            let m = map_m.get(&p).cloned().unwrap_or(empty.clone());
            results.insert(p, (u, l, m));
        }
        results
    }
}
