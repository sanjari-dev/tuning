use crate::gpu_indicators::GpuIndicatorHelper;
use arrow::array::{Array, Float64Array};
use arrow::datatypes::{DataType, Field};
use std::sync::Arc;
use rayon::prelude::*;

fn std_dev_sample(data: &[f64]) -> f64 {
    let len = data.len();
    if len < 2 { return 0.0; }

    let sum: f64 = data.iter().sum();
    let mean = sum / len as f64;
    let variance = data.iter().map(|value| {
        let diff = mean - value;
        diff * diff
    }).sum::<f64>() / (len - 1) as f64;

    variance.sqrt()
}

fn calculate_tr(high: f64, low: f64, prev_close: f64) -> f64 {
    let hl = high - low;
    let h_pc = (high - prev_close).abs();
    let l_pc = (low - prev_close).abs();
    hl.max(h_pc).max(l_pc)
}

pub fn calculate_sma(data: &[f64], period: usize) -> Vec<Option<f64>> {
    let len = data.len();
    let mut result = vec![None; len];
    if period == 0 || len < period { return result; }

    let mut sum: f64 = data.iter().take(period).sum();
    result[period - 1] = Some(sum / period as f64);

    for i in period..len {
        sum += data[i];
        sum -= data[i - period];
        result[i] = Some(sum / period as f64);
    }
    result
}

pub fn calculate_ema(data: &[f64], period: usize) -> Vec<Option<f64>> {
    let len = data.len();
    let mut result = vec![None; len];
    if period == 0 || len < period { return result; }

    let k = 2.0 / (period as f64 + 1.0);
    let sum: f64 = data.iter().take(period).sum();
    let mut prev_ema = sum / period as f64;
    result[period - 1] = Some(prev_ema);

    for i in period..len {
        let curr = (data[i] * k) + (prev_ema * (1.0 - k));
        result[i] = Some(curr);
        prev_ema = curr;
    }
    result
}

pub fn calculate_wma(data: &[f64], period: usize) -> Vec<Option<f64>> {
    let len = data.len();
    let mut result = Vec::with_capacity(len);
    if period == 0 || len < period {
        result.resize(len, None);
        return result;
    }

    let weight_sum = (period * (period + 1)) as f64 / 2.0;

    for _ in 0..period - 1 {
        result.push(None);
    }

    for i in (period - 1)..len {
        let start_idx = i + 1 - period;
        let window = &data[start_idx..=i];
        let mut numerator: f64 = 0.0;
        for (idx, &val) in window.iter().enumerate() {
            numerator += val * (idx + 1) as f64;
        }
        result.push(Some(numerator / weight_sum));
    }
    result
}

pub fn calculate_vwma(prices: &[f64], volumes: &[f64], period: usize) -> Vec<Option<f64>> {
    let len = prices.len();
    let mut result = vec![None; len];
    if period == 0 || len < period || len != volumes.len() { return result; }

    let mut sum_pv = 0.0;
    let mut sum_v = 0.0;

    for i in 0..period {
        sum_pv += prices[i] * volumes[i];
        sum_v += volumes[i];
    }
    if sum_v != 0.0 { result[period - 1] = Some(sum_pv / sum_v); }

    for i in period..len {
        sum_pv += (prices[i] * volumes[i]) - (prices[i - period] * volumes[i - period]);
        sum_v += volumes[i] - volumes[i - period];
        if sum_v != 0.0 { result[i] = Some(sum_pv / sum_v); }
    }
    result
}

pub fn calculate_hma(data: &[f64], period: usize) -> Vec<Option<f64>> {
    let len = data.len();
    if period == 0 || len < period { return vec![None; len]; }

    let half_n = period / 2;
    let sqrt_n = ((period as f64).sqrt().round() as usize).max(1);
    let wma_half = calculate_wma(data, half_n);
    let wma_full = calculate_wma(data, period);
    let start_calc = period - 1;
    let mut diff_series = Vec::with_capacity(len - start_calc);

    for i in start_calc..len {
        if let (Some(h), Some(f)) = (wma_half[i], wma_full[i]) {
            diff_series.push((2.0 * h) - f);
        } else {
            diff_series.push(0.0);
        }
    }

    let hma_raw = calculate_wma(&diff_series, sqrt_n);
    let mut result = vec![None; len];
    let total_offset = start_calc;

    for (i, val) in hma_raw.into_iter().enumerate() {
        if i + total_offset < len {
            result[i + total_offset] = val;
        }
    }
    result
}

pub fn calculate_smma(data: &[f64], period: usize) -> Vec<Option<f64>> {
    let len = data.len();
    let mut result = vec![None; len];
    if period == 0 || len < period { return result; }

    let sum: f64 = data.iter().take(period).sum();
    let mut prev = sum / period as f64;
    result[period - 1] = Some(prev);

    for i in period..len {
        let curr = (prev * (period as f64 - 1.0) + data[i]) / period as f64;
        result[i] = Some(curr);
        prev = curr;
    }
    result
}

pub fn calculate_dema(data: &[f64], period: usize) -> Vec<Option<f64>> {
    let len = data.len();
    let ema1 = calculate_ema(data, period);
    let valid_start = period - 1;
    if valid_start >= len { return vec![None; len]; }

    let mut ema1_raw = vec![0.0; len];
    for i in valid_start..len {
        if let Some(v) = ema1[i] { ema1_raw[i] = v; }
    }

    let mut result = vec![None; len];
    if valid_start + period > len { return result; }

    let k = 2.0 / (period as f64 + 1.0);
    let mut sum_ema1 = 0.0;
    for i in 0..period {
        sum_ema1 += ema1_raw[valid_start + i];
    }
    let mut prev_ema2 = sum_ema1 / period as f64;

    let idx_first = valid_start + period - 1;
    if idx_first < len {
        if let Some(e1) = ema1[idx_first] {
            result[idx_first] = Some(2.0 * e1 - prev_ema2);
        }
    }

    for i in (idx_first + 1)..len {
        if let Some(e1_curr) = ema1[i] {
            let curr_ema2 = (e1_curr * k) + (prev_ema2 * (1.0 - k));
            result[i] = Some(2.0 * e1_curr - curr_ema2);
            prev_ema2 = curr_ema2;
        }
    }
    result
}

pub fn calculate_tema(data: &[f64], period: usize) -> Vec<Option<f64>> {
    let len = data.len();
    if period == 0 || len < period { return vec![None; len]; }

    let calc_next_ema = |prev_series: &[Option<f64>]| -> Vec<Option<f64>> {
        let start_idx = prev_series.iter().position(|x| x.is_some()).unwrap_or(len);
        if len - start_idx < period { return vec![None; len]; }

        let valid_data: Vec<f64> = prev_series.iter().skip(start_idx).map(|x| x.unwrap_or(0.0)).collect();
        let next_ema_short = calculate_ema(&valid_data, period);

        let mut next_ema_full = vec![None; start_idx];
        next_ema_full.extend(next_ema_short);
        if next_ema_full.len() < len { next_ema_full.resize(len, None); }
        next_ema_full
    };

    let ema1 = calculate_ema(data, period);
    let ema2 = calc_next_ema(&ema1);
    let ema3 = calc_next_ema(&ema2);

    let mut result = vec![None; len];
    for i in 0..len {
        if let (Some(e1), Some(e2), Some(e3)) = (ema1[i], ema2[i], ema3[i]) {
            result[i] = Some(3.0 * e1 - 3.0 * e2 + e3);
        }
    }
    result
}

pub fn calculate_rsi(data: &[f64], period: usize) -> Vec<Option<f64>> {
    let len = data.len();
    let mut result = vec![None; len];
    if period == 0 || len <= period { return result; }

    let mut gains = 0.0;
    let mut losses = 0.0;

    for i in 1..=period {
        let change = data[i] - data[i - 1];
        if change > 0.0 { gains += change; } else { losses -= change; }
    }

    let mut avg_gain = gains / period as f64;
    let mut avg_loss = losses / period as f64;

    let calc = |g: f64, l: f64| -> f64 {
        if l == 0.0 { 100.0 } else { 100.0 - (100.0 / (1.0 + g / l)) }
    };
    result[period] = Some(calc(avg_gain, avg_loss));

    for i in (period + 1)..len {
        let change = data[i] - data[i - 1];
        let (g, l) = if change > 0.0 { (change, 0.0) } else { (0.0, -change) };
        avg_gain = ((avg_gain * (period as f64 - 1.0)) + g) / period as f64;
        avg_loss = ((avg_loss * (period as f64 - 1.0)) + l) / period as f64;
        result[i] = Some(calc(avg_gain, avg_loss));
    }
    result
}

pub fn calculate_cci(high: &[f64], low: &[f64], close: &[f64], period: usize) -> Vec<Option<f64>> {
    let len = close.len();
    let mut result = vec![None; len];
    if period == 0 || len < period { return result; }

    let mut tp = vec![0.0; len];
    for i in 0..len { tp[i] = (high[i] + low[i] + close[i]) / 3.0; }

    let sma_tp = calculate_sma(&tp, period);

    for i in (period - 1)..len {
        if let Some(ma) = sma_tp[i] {
            let mut mean_dev = 0.0;
            for j in 0..period {
                let idx = i.saturating_sub(j);
                mean_dev += (tp[idx] - ma).abs();
            }
            mean_dev /= period as f64;
            if mean_dev == 0.0 { result[i] = Some(0.0); }
            else { result[i] = Some((tp[i] - ma) / (0.015 * mean_dev)); }
        }
    }
    result
}

pub fn calculate_momentum(data: &[f64], period: usize) -> Vec<Option<f64>> {
    let len = data.len();
    let mut result = vec![None; len];
    if period == 0 { return result; }
    if len > period {
        for i in period..len { result[i] = Some(data[i] - data[i - period]); }
    }
    result
}

pub fn calculate_roc(data: &[f64], period: usize) -> Vec<Option<f64>> {
    let len = data.len();
    let mut result = vec![None; len];
    if period == 0 { return result; }
    if len > period {
        for i in period..len {
            let prev = data[i - period];
            if prev != 0.0 { result[i] = Some(((data[i] - prev) / prev) * 100.0); }
        }
    }
    result
}

pub fn calculate_williams_r(high: &[f64], low: &[f64], close: &[f64], period: usize) -> Vec<Option<f64>> {
    let len = close.len();
    let mut result = vec![None; len];

    if period == 0 || len < period { return result; }

    for i in (period - 1)..len {
        let start_idx = (i + 1).saturating_sub(period);
        let slice_h = &high[start_idx..=i];
        let slice_l = &low[start_idx..=i];

        let hh = slice_h.iter().fold(f64::MIN, |a, &b| a.max(b));
        let ll = slice_l.iter().fold(f64::MAX, |a, &b| a.min(b));

        if hh == ll {
            result[i] = Some(0.0);
        } else {
            result[i] = Some(((hh - close[i]) / (hh - ll)) * -100.0);
        }
    }
    result
}

pub fn calculate_atr(high: &[f64], low: &[f64], close: &[f64], period: usize) -> Vec<Option<f64>> {
    let len = close.len();
    let mut result = vec![None; len];
    if period == 0 || len <= period { return result; }

    let mut tr = vec![0.0; len];
    tr[0] = high[0] - low[0];
    for i in 1..len { tr[i] = calculate_tr(high[i], low[i], close[i - 1]); }

    let mut sum_tr = 0.0;
    for i in 0..period { sum_tr += tr[i]; }
    let mut prev_atr = sum_tr / period as f64;
    result[period - 1] = Some(prev_atr);

    for i in period..len {
        let curr = ((prev_atr * (period as f64 - 1.0)) + tr[i]) / period as f64;
        result[i] = Some(curr);
        prev_atr = curr;
    }
    result
}

pub fn calculate_adx_full(high: &[f64], low: &[f64], close: &[f64], period: usize) -> (Vec<Option<f64>>, Vec<Option<f64>>, Vec<Option<f64>>) {
    let len = close.len();
    let mut adx_res = vec![None; len];
    let mut pdi_res = vec![None; len];
    let mut mdi_res = vec![None; len];

    if period == 0 || len < period * 2 { return (adx_res, pdi_res, mdi_res); }

    let mut tr = vec![0.0; len];
    let mut dm_plus = vec![0.0; len];
    let mut dm_minus = vec![0.0; len];

    for i in 1..len {
        tr[i] = calculate_tr(high[i], low[i], close[i - 1]);
        let up = high[i] - high[i - 1];
        let down = low[i - 1] - low[i];
        if up > down && up > 0.0 { dm_plus[i] = up; }
        if down > up && down > 0.0 { dm_minus[i] = down; }
    }

    let mut sm_tr = 0.0;
    let mut sm_dm_p = 0.0;
    let mut sm_dm_m = 0.0;
    for i in 1..=period {
        sm_tr += tr[i]; sm_dm_p += dm_plus[i]; sm_dm_m += dm_minus[i];
    }

    let mut dx_vals = vec![0.0; len];

    for i in (period + 1)..len {
        sm_tr = sm_tr - (sm_tr / period as f64) + tr[i];
        sm_dm_p = sm_dm_p - (sm_dm_p / period as f64) + dm_plus[i];
        sm_dm_m = sm_dm_m - (sm_dm_m / period as f64) + dm_minus[i];

        let di_p = 100.0 * sm_dm_p / sm_tr;
        let di_m = 100.0 * sm_dm_m / sm_tr;

        pdi_res[i] = Some(di_p);
        mdi_res[i] = Some(di_m);

        let sum_di = di_p + di_m;
        if sum_di != 0.0 {
            dx_vals[i] = 100.0 * (di_p - di_m).abs() / sum_di;
        }
    }

    let start_adx = period * 2;
    if start_adx >= len { return (adx_res, pdi_res, mdi_res); }

    let mut sum_dx = 0.0;
    for i in (period + 1)..=start_adx { sum_dx += dx_vals[i]; }
    let mut prev_adx = sum_dx / period as f64;
    adx_res[start_adx] = Some(prev_adx);

    for i in (start_adx + 1)..len {
        let curr = ((prev_adx * (period as f64 - 1.0)) + dx_vals[i]) / period as f64;
        adx_res[i] = Some(curr);
        prev_adx = curr;
    }

    (adx_res, pdi_res, mdi_res)
}

pub fn calculate_trix(data: &[f64], period: usize) -> Vec<Option<f64>> {
    let len = data.len();
    if period == 0 || len < period * 3 { return vec![None; len]; }

    let ema1 = calculate_ema(data, period);
    let v1: Vec<f64> = ema1.iter().map(|x| x.unwrap_or(0.0)).collect();
    let ema2 = calculate_ema(&v1, period);
    let v2: Vec<f64> = ema2.iter().map(|x| x.unwrap_or(0.0)).collect();
    let ema3 = calculate_ema(&v2, period);

    let mut result = vec![None; len];
    for i in 1..len {
        if let (Some(curr), Some(prev)) = (ema3[i], ema3[i-1]) {
            if prev != 0.0 { result[i] = Some((curr - prev) / prev * 100.0); }
        }
    }
    result
}

pub fn calculate_donchian(high: &[f64], low: &[f64], period: usize) -> (Vec<Option<f64>>, Vec<Option<f64>>, Vec<Option<f64>>) {
    let len = high.len();
    let mut upper = vec![None; len];
    let mut lower = vec![None; len];
    let mut middle = vec![None; len];
    if period == 0 || len < period { return (upper, lower, middle); }

    for i in (period - 1)..len {
        let start_idx = (i + 1).saturating_sub(period);
        let h_slice = &high[start_idx..=i];
        let l_slice = &low[start_idx..=i];
        let hh = h_slice.iter().fold(f64::MIN, |a, &b| a.max(b));
        let ll = l_slice.iter().fold(f64::MAX, |a, &b| a.min(b));
        upper[i] = Some(hh);
        lower[i] = Some(ll);
        middle[i] = Some((hh + ll) / 2.0);
    }
    (upper, lower, middle)
}

pub fn calculate_envelope_full(data: &[f64], period: usize) -> (Vec<Option<f64>>, Vec<Option<f64>>, Vec<Option<f64>>) {
    let sma = calculate_sma(data, period);
    let upper = sma.iter().map(|opt| opt.map(|v| v * 1.025)).collect();
    let lower = sma.iter().map(|opt| opt.map(|v| v * 0.975)).collect();
    (upper, lower, sma)
}

pub fn calculate_obv(close: &[f64], volume: &[f64]) -> Vec<f64> {
    let len = close.len();
    if len == 0 { return vec![]; }
    let mut res = vec![0.0; len];
    let mut acc = 0.0;
    for i in 1..len {
        if close[i] > close[i-1] { acc += volume[i]; }
        else if close[i] < close[i-1] { acc -= volume[i]; }
        res[i] = acc;
    }
    res
}

pub fn calculate_cmf(high: &[f64], low: &[f64], close: &[f64], volume: &[f64], period: usize) -> Vec<Option<f64>> {
    let len = close.len();
    let mut result = vec![None; len];
    if period == 0 || len < period { return result; }

    let mut mfv = vec![0.0; len];
    for i in 0..len {
        let hl = high[i] - low[i];
        if hl != 0.0 {
            let mfm = ((close[i] - low[i]) - (high[i] - close[i])) / hl;
            mfv[i] = mfm * volume[i];
        }
    }

    for i in (period - 1)..len {
        let start_idx = (i + 1).saturating_sub(period);
        let sum_mfv: f64 = mfv[start_idx..=i].iter().sum();
        let sum_vol: f64 = volume[start_idx..=i].iter().sum();
        if sum_vol != 0.0 { result[i] = Some(sum_mfv / sum_vol); }
    }
    result
}

pub fn calculate_force_index(close: &[f64], volume: &[f64], period: usize) -> Vec<Option<f64>> {
    let len = close.len();
    if len == 0 { return vec![]; }
    let mut raw_force = vec![0.0; len];
    for i in 1..len {
        raw_force[i] = (close[i] - close[i-1]) * volume[i];
    }
    calculate_ema(&raw_force, period)
}

pub fn calculate_mfi(high: &[f64], low: &[f64], close: &[f64], volume: &[f64], period: usize) -> Vec<Option<f64>> {
    let len = close.len();
    let mut result = vec![None; len];
    if period == 0 || len <= period { return result; }

    let mut tp = vec![0.0; len];
    for i in 0..len { tp[i] = (high[i] + low[i] + close[i]) / 3.0; }

    let mut pos_flow = vec![0.0; len];
    let mut neg_flow = vec![0.0; len];

    for i in 1..len {
        let flow = tp[i] * volume[i];
        if tp[i] > tp[i-1] { pos_flow[i] = flow; }
        else if tp[i] < tp[i-1] { neg_flow[i] = flow; }
    }

    for i in period..len {
        let start_idx = (i + 1).saturating_sub(period);
        let sum_p: f64 = pos_flow[start_idx..=i].iter().sum();
        let sum_n: f64 = neg_flow[start_idx..=i].iter().sum();

        if sum_n == 0.0 { result[i] = Some(100.0); }
        else {
            let mr = sum_p / sum_n;
            result[i] = Some(100.0 - (100.0 / (1.0 + mr)));
        }
    }
    result
}

pub fn calculate_vortex(high: &[f64], low: &[f64], close: &[f64], period: usize) -> (Vec<Option<f64>>, Vec<Option<f64>>) {
    let len = close.len();
    let mut vi_plus = vec![None; len];
    let mut vi_minus = vec![None; len];
    if period == 0 || len <= period { return (vi_plus, vi_minus); }

    let mut vm_p = vec![0.0; len];
    let mut vm_m = vec![0.0; len];
    let mut tr = vec![0.0; len];

    for i in 1..len {
        vm_p[i] = (high[i] - low[i-1]).abs();
        vm_m[i] = (low[i] - high[i-1]).abs();
        tr[i] = calculate_tr(high[i], low[i], close[i-1]);
    }

    for i in period..len {
        let start_idx = (i + 1).saturating_sub(period);
        let sum_vp: f64 = vm_p[start_idx..=i].iter().sum();
        let sum_vm: f64 = vm_m[start_idx..=i].iter().sum();
        let sum_tr: f64 = tr[start_idx..=i].iter().sum();
        if sum_tr != 0.0 {
            vi_plus[i] = Some(sum_vp / sum_tr);
            vi_minus[i] = Some(sum_vm / sum_tr);
        }
    }
    (vi_plus, vi_minus)
}

pub fn calculate_aroon(high: &[f64], low: &[f64], period: usize) -> (Vec<Option<f64>>, Vec<Option<f64>>) {
    let len = high.len();
    let mut up_vec = vec![None; len];
    let mut down_vec = vec![None; len];
    if period == 0 || len < period { return (up_vec, down_vec); }

    let calc_aroon_val = |slice: &[f64], is_high: bool| -> f64 {
        let init_val = if is_high { f64::MIN } else { f64::MAX };
        let (idx, _) = slice.iter().enumerate().fold((0, init_val), |(i_ex, v_ex), (i, &v)| {
            let update = if is_high { v >= v_ex } else { v <= v_ex };
            if update { (i, v) } else { (i_ex, v_ex) }
        });

        let days_since = (period as f64 - 1.0) - idx as f64;
        ((period as f64 - days_since) / period as f64) * 100.0
    };

    for i in (period - 1)..len {
        let start_idx = (i + 1).saturating_sub(period);
        let slice_h = &high[start_idx..=i];
        let slice_l = &low[start_idx..=i];

        up_vec[i] = Some(calc_aroon_val(slice_h, true));
        down_vec[i] = Some(calc_aroon_val(slice_l, false));
    }
    (up_vec, down_vec)
}

pub fn calculate_aroon_osc(up: &[Option<f64>], down: &[Option<f64>]) -> Vec<Option<f64>> {
    up.iter().zip(down.iter()).map(|(u, d)| {
        match (u, d) {
            (Some(u_val), Some(d_val)) => Some(u_val - d_val),
            _ => None,
        }
    }).collect()
}

pub fn calculate_cmo(data: &[f64], period: usize) -> Vec<Option<f64>> {
    let len = data.len();
    let mut result = vec![None; len];
    if period == 0 || len <= period { return result; }

    let mut up_sum = 0.0;
    let mut down_sum = 0.0;
    for i in 1..=period {
        if data[i] > data[i-1] { up_sum += data[i] - data[i-1]; }
        else { down_sum += data[i-1] - data[i]; }
    }

    let calc = |u, d| (u - d) / (u + d) * 100.0;
    if up_sum + down_sum != 0.0 { result[period] = Some(calc(up_sum, down_sum)); }

    for i in (period + 1)..len {
        let diff_new = data[i] - data[i-1];
        if diff_new > 0.0 { up_sum += diff_new; } else { down_sum -= diff_new; }

        let diff_old = data[i-period] - data[i-period-1];
        if diff_old > 0.0 { up_sum -= diff_old; } else { down_sum += diff_old; }
        if up_sum + down_sum != 0.0 { result[i] = Some(calc(up_sum, down_sum)); }
    }
    result
}

pub fn calculate_std_dev(data: &[f64], period: usize) -> Vec<Option<f64>> {
    let len = data.len();
    let mut result = vec![None; len];
    if period == 0 || len < period { return result; }
    for i in (period - 1)..len {
        let start_idx = (i + 1).saturating_sub(period);
        let slice = &data[start_idx..=i];
        result[i] = Some(std_dev_sample(slice));
    }
    result
}

pub fn calculate_variance(data: &[f64], period: usize) -> Vec<Option<f64>> {
    let len = data.len();
    let mut result = vec![None; len];
    if period == 0 || len < period { return result; }
    for i in (period - 1)..len {
        let start_idx = (i + 1).saturating_sub(period);
        let slice = &data[start_idx..=i];
        let sd = std_dev_sample(slice);
        result[i] = Some(sd * sd);
    }
    result
}

pub fn calculate_median(data: &[f64], period: usize) -> Vec<Option<f64>> {
    let len = data.len();
    let mut result = vec![None; len];
    if period == 0 || len < period { return result; }
    for i in (period - 1)..len {
        let start_idx = (i + 1).saturating_sub(period);
        let mut window = data[start_idx..=i].to_vec();
        window.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let mid = window.len() / 2;
        if window.len() % 2 == 0 {
            result[i] = Some((window[mid-1] + window[mid]) / 2.0);
        } else {
            result[i] = Some(window[mid]);
        }
    }
    result
}

pub fn calculate_fcb(high: &[f64], low: &[f64], period: usize) -> (Vec<Option<f64>>, Vec<Option<f64>>) {
    let len = high.len();
    let mut upper = vec![None; len];
    let mut lower = vec![None; len];

    if period == 0 || len < period { return (upper, lower); }

    for i in period..len {
        let idx_prev = i - 1;
        let start_idx = (idx_prev + 1).saturating_sub(period);
        let h_slice = &high[start_idx..=idx_prev];
        let l_slice = &low[start_idx..=idx_prev];
        let hh = h_slice.iter().fold(f64::MIN, |a, &b| a.max(b));
        let ll = l_slice.iter().fold(f64::MAX, |a, &b| a.min(b));

        upper[i] = Some(hh);
        lower[i] = Some(ll);
    }
    (upper, lower)
}

struct LocalBuilder {
    fields: Vec<Field>,
    cols: Vec<Arc<dyn Array>>,
    offset: usize,
}

impl LocalBuilder {
    fn new(offset: usize) -> Self {
        Self {
            fields: Vec::new(),
            cols: Vec::new(),
            offset,
        }
    }

    fn add_arc(&mut self, name: &str, arr: Arc<dyn Array>) {
        self.fields.push(Field::new(name, DataType::Float64, true));
        if self.offset < arr.len() {
            self.cols.push(arr.slice(self.offset, arr.len() - self.offset));
        } else {
            self.cols.push(Arc::new(Float64Array::from(Vec::<Option<f64>>::new())));
        }
    }

    fn add_vec(&mut self, name: &str, data: Vec<Option<f64>>) {
        self.fields.push(Field::new(name, DataType::Float64, true));
        let sliced_data = if self.offset < data.len() {
            data[self.offset..].to_vec()
        } else {
            Vec::new()
        };
        self.cols.push(Arc::new(Float64Array::from(sliced_data)));
    }
}

pub fn append_indicators(
    fields: &mut Vec<Field>,
    columns: &mut Vec<Arc<dyn Array>>,
    _open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    volumes: &[f64],
    min_period: usize,
    max_period: usize,
    offset: usize,
    gpu_helper: Option<&GpuIndicatorHelper>,
) {
    let obv_vals = calculate_obv(close, volumes);
    fields.push(Field::new("obv_line", DataType::Float64, false));
    columns.push(Arc::new(Float64Array::from(obv_vals[offset..].to_vec())));

    let (gpu_smas, gpu_wmas, gpu_std_pair, gpu_envs) = if let Some(gpu) = gpu_helper {
        let range = min_period..=max_period;
        (
            Some(gpu.compute_all_smas(range.clone())),
            Some(gpu.compute_all_wmas(range.clone())),
            Some(gpu.compute_all_std_devs(range.clone())),
            Some(gpu.compute_all_envelopes(range.clone()))
        )
    } else {
        (None, None, None, None)
    };

    let (gpu_std_map, gpu_var_map) = if let Some((s, v)) = gpu_std_pair {
        (Some(s), Some(v))
    } else {
        (None, None)
    };

    let high_ref = high.to_vec();
    let low_ref = low.to_vec();
    let close_ref = close.to_vec();
    let vol_ref = volumes.to_vec();
    let obv_ref = obv_vals;
    let results: Vec<(Vec<Field>, Vec<Arc<dyn Array>>)> = (min_period..=max_period)
        .into_par_iter()
        .map(|period| {
            let mut builder = LocalBuilder::new(offset);

            if let Some(arr) = gpu_smas.as_ref().and_then(|m| m.get(&period)) {
                builder.add_arc(&format!("sma_{}", period), arr.clone());
            } else {
                builder.add_vec(&format!("sma_{}", period), calculate_sma(&close_ref, period));
            }

            builder.add_vec(&format!("ema_{}", period), calculate_ema(&close_ref, period));

            if let Some(arr) = gpu_wmas.as_ref().and_then(|m| m.get(&period)) {
                builder.add_arc(&format!("wma_{}", period), arr.clone());
            } else {
                builder.add_vec(&format!("wma_{}", period), calculate_wma(&close_ref, period));
            }

            builder.add_vec(&format!("vwma_{}", period), calculate_vwma(&close_ref, &vol_ref, period));
            builder.add_vec(&format!("hma_{}", period), calculate_hma(&close_ref, period));
            builder.add_vec(&format!("smma_{}", period), calculate_smma(&close_ref, period));
            builder.add_vec(&format!("dema_{}", period), calculate_dema(&close_ref, period));
            builder.add_vec(&format!("tema_{}", period), calculate_tema(&close_ref, period));
            builder.add_vec(&format!("rsi_{}", period), calculate_rsi(&close_ref, period));
            builder.add_vec(&format!("cci_{}", period), calculate_cci(&high_ref, &low_ref, &close_ref, period));
            builder.add_vec(&format!("mom_{}", period), calculate_momentum(&close_ref, period));
            builder.add_vec(&format!("roc_{}", period), calculate_roc(&close_ref, period));
            builder.add_vec(&format!("wpr_{}", period), calculate_williams_r(&high_ref, &low_ref, &close_ref, period));
            builder.add_vec(&format!("atr_{}", period), calculate_atr(&high_ref, &low_ref, &close_ref, period));

            let (adx, pdi, mdi) = calculate_adx_full(&high_ref, &low_ref, &close_ref, period);
            builder.add_vec(&format!("adx_{}", period), adx);
            builder.add_vec(&format!("pdi_{}", period), pdi);
            builder.add_vec(&format!("mdi_{}", period), mdi);
            builder.add_vec(&format!("trix_{}", period), calculate_trix(&close_ref, period));
            builder.add_vec(&format!("sm_rsi_{}", period), calculate_rsi(&close_ref, period));

            let (don_u, don_l, don_m) = calculate_donchian(&high_ref, &low_ref, period);
            builder.add_vec(&format!("donchian_upper_{}", period), don_u);
            builder.add_vec(&format!("donchian_lower_{}", period), don_l);
            builder.add_vec(&format!("donchian_middle_{}", period), don_m);

            if let Some((u, l, m)) = gpu_envs.as_ref().and_then(|map| map.get(&period)) {
                builder.add_arc(&format!("env_upper_{}", period), u.clone());
                builder.add_arc(&format!("env_lower_{}", period), l.clone());
                builder.add_arc(&format!("env_middle_{}", period), m.clone());
            } else {
                let (u, l, m) = calculate_envelope_full(&close_ref, period);
                builder.add_vec(&format!("env_upper_{}", period), u);
                builder.add_vec(&format!("env_lower_{}", period), l);
                builder.add_vec(&format!("env_middle_{}", period), m);
            }

            builder.add_vec(&format!("fi_{}", period), calculate_force_index(&close_ref, &vol_ref, period));
            builder.add_vec(&format!("cmf_{}", period), calculate_cmf(&high_ref, &low_ref, &close_ref, &vol_ref, period));
            builder.add_vec(&format!("mfi_{}", period), calculate_mfi(&high_ref, &low_ref, &close_ref, &vol_ref, period));

            let (vi_p, vi_m) = calculate_vortex(&high_ref, &low_ref, &close_ref, period);
            builder.add_vec(&format!("vi_plus_{}", period), vi_p);
            builder.add_vec(&format!("vi_minus_{}", period), vi_m);

            let (ar_u, ar_d) = calculate_aroon(&high_ref, &low_ref, period);
            builder.add_vec(&format!("aroon_up_{}", period), ar_u.clone());
            builder.add_vec(&format!("aroon_down_{}", period), ar_d.clone());
            builder.add_vec(&format!("aroon_osc_{}", period), calculate_aroon_osc(&ar_u, &ar_d));
            builder.add_vec(&format!("cmo_{}", period), calculate_cmo(&close_ref, period));

            if let (Some(s_map), Some(v_map)) = (&gpu_std_map, &gpu_var_map) {
                if let (Some(d_s), Some(d_v)) = (s_map.get(&period), v_map.get(&period)) {
                    builder.add_arc(&format!("std_dev_{}", period), d_s.clone());
                    builder.add_arc(&format!("variance_{}", period), d_v.clone());
                } else {
                    builder.add_vec(&format!("std_dev_{}", period), calculate_std_dev(&close_ref, period));
                    builder.add_vec(&format!("variance_{}", period), calculate_variance(&close_ref, period));
                }
            } else {
                builder.add_vec(&format!("std_dev_{}", period), calculate_std_dev(&close_ref, period));
                builder.add_vec(&format!("variance_{}", period), calculate_variance(&close_ref, period));
            }

            builder.add_vec(&format!("median_{}", period), calculate_median(&close_ref, period));

            let (fcb_u, fcb_l) = calculate_fcb(&high_ref, &low_ref, period);
            builder.add_vec(&format!("fcb_upper_{}", period), fcb_u);
            builder.add_vec(&format!("fcb_lower_{}", period), fcb_l);
            builder.add_vec(&format!("obv_sma_{}", period), calculate_sma(&obv_ref, period));

            (builder.fields, builder.cols)
        })
        .collect();

    for (f, c) in results {
        fields.extend(f);
        columns.extend(c);
    }
}