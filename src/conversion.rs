use arrow::array::{
    Array, Decimal128Array, StringArray, TimestampSecondArray, UInt32Array, UInt64Array, UInt8Array,
};
use arrow::datatypes::{DataType, Field, TimeUnit};
use chrono::{DateTime, Datelike, Timelike};
use clickhouse_rs::types::Block;
use rust_decimal::prelude::*;
use std::error::Error;
use std::sync::Arc;

const TARGET_SCALE: u32 = 5;

pub struct BatchMathData {
    pub opens: Vec<f64>,
    pub highs: Vec<f64>,
    pub lows: Vec<f64>,
    pub closes: Vec<f64>,
    pub volumes: Vec<f64>,
}

pub fn parse_block(
    block: &Block,
) -> Result<(Vec<Arc<dyn Array>>, BatchMathData, Vec<Field>), Box<dyn Error>> {
    let row_count = block.rows().count();
    let mut timestamps: Vec<i64> = Vec::with_capacity(row_count);
    let mut instruments: Vec<String> = Vec::with_capacity(row_count);
    let mut timeframes: Vec<String> = Vec::with_capacity(row_count);
    let mut opens_arr: Vec<i128> = Vec::with_capacity(row_count);
    let mut highs_arr: Vec<i128> = Vec::with_capacity(row_count);
    let mut lows_arr: Vec<i128> = Vec::with_capacity(row_count);
    let mut closes_arr: Vec<i128> = Vec::with_capacity(row_count);
    let mut min_spreads_arr: Vec<i128> = Vec::with_capacity(row_count);
    let mut max_spreads_arr: Vec<i128> = Vec::with_capacity(row_count);
    let mut avg_spreads_arr: Vec<i128> = Vec::with_capacity(row_count);
    let mut vwaps_arr: Vec<i128> = Vec::with_capacity(row_count);
    let mut tick_counts: Vec<u32> = Vec::with_capacity(row_count);
    let mut bid_vols: Vec<u64> = Vec::with_capacity(row_count);
    let mut ask_vols: Vec<u64> = Vec::with_capacity(row_count);
    let mut minutes: Vec<u8> = Vec::with_capacity(row_count);
    let mut hours: Vec<u8> = Vec::with_capacity(row_count);
    let mut days_of_week: Vec<u8> = Vec::with_capacity(row_count);
    let mut math_data = BatchMathData {
        opens: Vec::with_capacity(row_count),
        highs: Vec::with_capacity(row_count),
        lows: Vec::with_capacity(row_count),
        closes: Vec::with_capacity(row_count),
        volumes: Vec::with_capacity(row_count),
    };

    let to_decimal_i128 = |val: f64| -> i128 {
        Decimal::from_f64(val)
            .unwrap_or(Decimal::ZERO)
            .round_dp(TARGET_SCALE)
            .mantissa()
    };

    for row in block.rows() {
        let ts: u32 = row.get("ts")?;
        timestamps.push(ts as i64);

        if let Some(dt) = DateTime::from_timestamp(ts as i64, 0) {
            minutes.push(dt.minute() as u8);
            hours.push(dt.hour() as u8);
            days_of_week.push(dt.weekday().number_from_monday() as u8);
        } else {
            minutes.push(0);
            hours.push(0);
            days_of_week.push(0);
        }

        instruments.push(row.get("inst")?);
        timeframes.push(row.get("tf")?);
        tick_counts.push(row.get("tick_count")?);

        let bv: u64 = row.get("total_bid_volume")?;
        let av: u64 = row.get("total_ask_volume")?;
        bid_vols.push(bv);
        ask_vols.push(av);
        math_data.volumes.push((bv + av) as f64);

        let open: f64 = row.get("open")?;
        let high: f64 = row.get("high")?;
        let low: f64 = row.get("low")?;
        let close: f64 = row.get("close")?;

        math_data.opens.push(open);
        math_data.highs.push(high);
        math_data.lows.push(low);
        math_data.closes.push(close);

        opens_arr.push(to_decimal_i128(open));
        highs_arr.push(to_decimal_i128(high));
        lows_arr.push(to_decimal_i128(low));
        closes_arr.push(to_decimal_i128(close));

        let min_spr: f64 = row.get("min_spr")?;
        let max_spr: f64 = row.get("max_spr")?;
        let avg_spr: f64 = row.get("avg_spr")?;
        let vwap: f64 = row.get("vwap")?;

        min_spreads_arr.push(to_decimal_i128(min_spr));
        max_spreads_arr.push(to_decimal_i128(max_spr));
        avg_spreads_arr.push(to_decimal_i128(avg_spr));
        vwaps_arr.push(to_decimal_i128(vwap));
    }

    let timezone_utc = "+00:00";
    let decimal_type = DataType::Decimal128(18, 5);
    let make_dec_arr = |data: Vec<i128>| -> Arc<dyn Array> {
        Arc::new(Decimal128Array::from(data).with_data_type(decimal_type.clone()))
    };

    let fields = vec![
        Field::new(
            "timestamp",
            DataType::Timestamp(TimeUnit::Second, Some(timezone_utc.into())),
            false,
        ),
        Field::new("instrument", DataType::Utf8, false),
        Field::new("timeframe", DataType::Utf8, false),
        Field::new("open", decimal_type.clone(), false),
        Field::new("high", decimal_type.clone(), false),
        Field::new("low", decimal_type.clone(), false),
        Field::new("close", decimal_type.clone(), false),
        Field::new("tick_count", DataType::UInt32, false),
        Field::new("min_spread", decimal_type.clone(), false),
        Field::new("max_spread", decimal_type.clone(), false),
        Field::new("avg_spread", decimal_type.clone(), false),
        Field::new("total_bid_volume", DataType::UInt64, false),
        Field::new("total_ask_volume", DataType::UInt64, false),
        Field::new("vwap", decimal_type.clone(), false),
        Field::new("time_minute_of_hour", DataType::UInt8, false),
        Field::new("time_hour_of_day", DataType::UInt8, false),
        Field::new("time_day_of_week", DataType::UInt8, false),
    ];

    let columns: Vec<Arc<dyn Array>> = vec![
        Arc::new(TimestampSecondArray::from(timestamps).with_timezone(timezone_utc)),
        Arc::new(StringArray::from(instruments)),
        Arc::new(StringArray::from(timeframes)),
        make_dec_arr(opens_arr),
        make_dec_arr(highs_arr),
        make_dec_arr(lows_arr),
        make_dec_arr(closes_arr),
        Arc::new(UInt32Array::from(tick_counts)),
        make_dec_arr(min_spreads_arr),
        make_dec_arr(max_spreads_arr),
        make_dec_arr(avg_spreads_arr),
        Arc::new(UInt64Array::from(bid_vols)),
        Arc::new(UInt64Array::from(ask_vols)),
        make_dec_arr(vwaps_arr),
        Arc::new(UInt8Array::from(minutes)),
        Arc::new(UInt8Array::from(hours)),
        Arc::new(UInt8Array::from(days_of_week)),
    ];

    Ok((columns, math_data, fields))
}
