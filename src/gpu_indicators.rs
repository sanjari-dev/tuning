use burn::backend::wgpu::{Wgpu, WgpuDevice};
use burn::tensor::{Tensor, TensorData};
use burn::tensor::module::conv1d;
use burn::tensor::ops::ConvOptions;
use burn::nn::pool::AvgPool1dConfig;
use burn::nn::PaddingConfig1d;

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

    fn pad_and_export(&self, tensor: Tensor<B, 3>, period: usize) -> Vec<Option<f64>> {
        let data_vec: Vec<f32> = tensor.into_data().to_vec().expect("Failed to download tensor");
        let mut result = Vec::with_capacity(self.length);

        for _ in 0..(period - 1) {
            result.push(None);
        }

        for val in data_vec {
            result.push(Some(val as f64));
        }

        while result.len() < self.length {
            result.push(None);
        }

        result.truncate(self.length);
        result
    }

    pub fn calculate_sma(&self, period: usize) -> Vec<Option<f64>> {
        let config = AvgPool1dConfig::new(period)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Valid);
        let pool = config.init();
        let result = pool.forward(self.close.clone());
        self.pad_and_export(result, period)
    }

    pub fn calculate_wma(&self, period: usize) -> Vec<Option<f64>> {
        let weight_sum = (period * (period + 1)) as f32 / 2.0;
        let mut weights: Vec<f32> = Vec::with_capacity(period);
        for i in 1..=period {
            weights.push(i as f32 / weight_sum);
        }
        let weight_tensor = Tensor::from_data(
            TensorData::new(weights, [1, 1, period]),
            &self.device
        );
        let options = ConvOptions {
            stride: [1],
            padding: [0],
            dilation: [1],
            groups: 1,
        };
        let result = conv1d(
            self.close.clone(),
            weight_tensor,
            None,
            options
        );
        self.pad_and_export(result, period)
    }

    pub fn calculate_std_dev(&self, period: usize) -> Vec<Option<f64>> {
        let config = AvgPool1dConfig::new(period)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Valid);
        let pool = config.init();
        let sma_x = pool.forward(self.close.clone());
        let sma_x2 = pool.forward(self.close_sq.clone());
        let variance = sma_x2.sub(sma_x.powf_scalar(2.0));
        let std_dev = variance.clamp_min(0.0).sqrt();
        self.pad_and_export(std_dev, period)
    }

    pub fn calculate_variance(&self, period: usize) -> Vec<Option<f64>> {
        let config = AvgPool1dConfig::new(period)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Valid);
        let pool = config.init();
        let sma_x = pool.forward(self.close.clone());
        let sma_x2 = pool.forward(self.close_sq.clone());
        let variance = sma_x2.sub(sma_x.powf_scalar(2.0));
        self.pad_and_export(variance, period)
    }

    pub fn calculate_envelope(&self, period: usize) -> (Vec<Option<f64>>, Vec<Option<f64>>, Vec<Option<f64>>) {
        let config = AvgPool1dConfig::new(period)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Valid);
        let pool = config.init();
        let sma = pool.forward(self.close.clone());
        let upper = sma.clone().mul_scalar(1.025);
        let lower = sma.clone().mul_scalar(0.975);

        (
            self.pad_and_export(upper, period),
            self.pad_and_export(lower, period),
            self.pad_and_export(sma, period)
        )
    }
}