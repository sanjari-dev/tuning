mod indicators;

use arrow::array::{
    Array, Decimal128Array, RecordBatch, StringArray, TimestampSecondArray,
    UInt32Array, UInt64Array, UInt8Array,
};
use arrow::datatypes::{DataType, Field, Schema, TimeUnit};
use arrow::csv::WriterBuilder;
use clickhouse_rs::Pool;
use dotenvy::dotenv;
use rust_decimal::prelude::*;
use std::env;
use std::error::Error;
use std::fs::File;
use std::str::FromStr;
use std::sync::Arc;
use chrono::{DateTime, Timelike, Datelike};

fn validate_input(input: &str, field_name: &str) -> Result<String, Box<dyn Error>> {
    if input.chars().all(|c| c.is_alphanumeric() || c == '-' || c == '_' || c == ' ' || c == ':') {
        Ok(input.to_string())
    } else {
        Err(format!("Invalid characters in {}: potentially unsafe.", field_name).into())
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    dotenv().ok();

    const TARGET_SCALE: u32 = 5;

    let db_host = env::var("CLICKHOUSE_HOST").expect("HOST not set");
    let db_port = env::var("CLICKHOUSE_PORT").unwrap_or_else(|_| "9000".to_string());
    let db_user = env::var("CLICKHOUSE_USER").unwrap_or_else(|_| "default".to_string());
    let db_pass = env::var("CLICKHOUSE_PASSWORD").unwrap_or_else(|_| "".to_string());
    let db_name = env::var("CLICKHOUSE_DB").expect("DB name not set");

    let q_instrument = validate_input(&env::var("QUERY_INSTRUMENT").unwrap_or_else(|_| "EURUSD".to_string()), "Instrument")?;
    let q_timeframe = validate_input(&env::var("QUERY_TIMEFRAME").unwrap_or_else(|_| "m1".to_string()), "Timeframe")?;
    let q_start = validate_input(&env::var("QUERY_START_DATE").expect("START DATE not set"), "Start Date")?;
    let q_end = validate_input(&env::var("QUERY_END_DATE").expect("END DATE not set"), "End Date")?;

    let db_url = format!("tcp://{}:{}@{}:{}/{}", db_user, db_pass, db_host, db_port, db_name);
    println!("Connecting via TCP to {}...", db_url);

    let pool = Pool::new(db_url);
    let mut client = pool.get_handle().await?;
    let sql_query = format!(r#"
        SELECT
            toUnixTimestamp(timestamp) as ts,
            toString(instrument) as inst,
            toString(timeframe) as tf,
            toString(open) as open,
            toString(high) as high,
            toString(low) as low,
            toString(close) as close,
            tick_count,
            toString(min_spread) as min_spr,
            toString(max_spread) as max_spr,
            toString(avg_spread) as avg_spr,
            total_bid_volume,
            total_ask_volume,
            toString(vwap) as vwap
        FROM candles
        WHERE
            instrument = '{}'
            AND timeframe = '{}'
            AND timestamp >= toDateTime('{}')
            AND timestamp <= toDateTime('{}')
        ORDER BY timestamp ASC
        LIMIT 1000
    "#, q_instrument, q_timeframe, q_start, q_end);

    println!("Executing query...");
    let block = client.query(&sql_query).fetch_all().await?;
    let row_count = block.rows().count();
    println!("Fetched {} rows.", row_count);

    if row_count == 0 {
        return Ok(());
    }

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
    let mut opens_f64: Vec<f64> = Vec::with_capacity(row_count);
    let mut highs_f64: Vec<f64> = Vec::with_capacity(row_count);
    let mut lows_f64: Vec<f64> = Vec::with_capacity(row_count);
    let mut closes_f64: Vec<f64> = Vec::with_capacity(row_count);
    let mut volumes_f64: Vec<f64> = Vec::with_capacity(row_count);

    for row in block.rows() {
        let ts = row.get::<u32, _>("ts")?;timestamps.push(ts as i64);
        if let Some(dt) = DateTime::from_timestamp(ts as i64, 0) {
            minutes.push(dt.minute() as u8);
            hours.push(dt.hour() as u8);
            days_of_week.push(dt.weekday().number_from_monday() as u8);
        } else {
            minutes.push(0); hours.push(0); days_of_week.push(0);
        }

        instruments.push(row.get::<String, _>("inst")?);
        timeframes.push(row.get::<String, _>("tf")?);
        tick_counts.push(row.get::<u32, _>("tick_count")?);

        let bv = row.get::<u64, _>("total_bid_volume")?;bid_vols.push(bv);
        let av = row.get::<u64, _>("total_ask_volume")?;ask_vols.push(av);
        let parse_arrow_dec = |s: &str| -> Result<i128, Box<dyn Error>> {
            let mut d = Decimal::from_str(s).unwrap_or(Decimal::ZERO);
            d.rescale(TARGET_SCALE);
            Ok(d.mantissa())
        };
        let parse_math_f64 = |s: &str| -> f64 {
            s.parse::<f64>().unwrap_or(0.0)
        };

        let open_str = row.get::<String, _>("open")?;
        let high_str = row.get::<String, _>("high")?;
        let low_str = row.get::<String, _>("low")?;
        let close_str = row.get::<String, _>("close")?;
        let min_spr_str = row.get::<String, _>("min_spr")?;
        let max_spr_str = row.get::<String, _>("max_spr")?;
        let avg_spr_str = row.get::<String, _>("avg_spr")?;
        let vwap_str = row.get::<String, _>("vwap")?;

        opens_arr.push(parse_arrow_dec(&open_str)?);
        highs_arr.push(parse_arrow_dec(&high_str)?);
        lows_arr.push(parse_arrow_dec(&low_str)?);
        closes_arr.push(parse_arrow_dec(&close_str)?);
        min_spreads_arr.push(parse_arrow_dec(&min_spr_str)?);
        max_spreads_arr.push(parse_arrow_dec(&max_spr_str)?);
        avg_spreads_arr.push(parse_arrow_dec(&avg_spr_str)?);
        vwaps_arr.push(parse_arrow_dec(&vwap_str)?);
        opens_f64.push(parse_math_f64(&open_str));
        highs_f64.push(parse_math_f64(&high_str));
        lows_f64.push(parse_math_f64(&low_str));
        closes_f64.push(parse_math_f64(&close_str));
        volumes_f64.push((bv + av) as f64);
    }

    let timezone_utc = "+00:00";
    let decimal_type = DataType::Decimal128(9, 5);
    let make_dec_arr = |data: Vec<i128>| -> Arc<dyn Array> {
        Arc::new(Decimal128Array::from(data).with_data_type(DataType::Decimal128(9, 5)))
    };

    let mut fields = vec![
        Field::new("timestamp", DataType::Timestamp(TimeUnit::Second, Some(timezone_utc.into())), false),
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

    let mut columns: Vec<Arc<dyn Array>> = vec![
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

    indicators::append_indicators(
        &mut fields,
        &mut columns,
        &opens_f64,
        &highs_f64,
        &lows_f64,
        &closes_f64,
        &volumes_f64,
        5,
        200
    );

    let schema = Schema::new(fields);
    let batch = RecordBatch::try_new(Arc::new(schema), columns)?;

    println!("Batch created successfully.");
    println!("Total Columns: {}", batch.num_columns());

    use arrow::compute::filter_record_batch;
    use arrow::array::BooleanArray;

    println!("Filtering NULL rows (warmup period)...");
    let num_rows = batch.num_rows();
    let mut keep_row = vec![true; num_rows];
    for col in batch.columns() {
        if col.null_count() > 0 {
            for i in 0..num_rows {
                if col.is_null(i) {
                    keep_row[i] = false;
                }
            }
        }
    }

    let predicate = BooleanArray::from(keep_row);
    let clean_batch = filter_record_batch(&batch, &predicate)?;
    println!("Clean Batch Rows: {} (Original: {})", clean_batch.num_rows(), num_rows);

    if clean_batch.num_rows() == 0 {
        eprintln!("WARNING: All rows were filtered out! This is expected if data length < max period (200).");
        return Ok(());
    }

    let output_file = "tuning_output.csv";
    println!("Exporting to CSV: {} ...", output_file);

    let file = File::create(output_file)?;

    let mut writer = WriterBuilder::new()
        .with_header(true)
        .build(file);

    writer.write(&clean_batch)?;

    println!("Done! Successfully exported data to '{}'.", output_file);

    Ok(())
}