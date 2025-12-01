mod config;
mod conversion;
mod indicators;
mod phase1;
mod utils;
mod repository;

use arrow::array::RecordBatch;
use arrow::csv::{ReaderBuilder, WriterBuilder};
use arrow::datatypes::Schema;
use futures_util::stream::StreamExt;
use std::error::Error;
use std::fs::{self, File};
use std::sync::Arc;
use std::time::Instant;

use config::AppConfig;
use conversion::parse_block;
use phase1::Phase1Cleaner;
use repository::ClickHouseRepository;
use utils::{filter_null_rows, get_tail};

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let start_app = Instant::now();

    const MAX_PERIOD: usize = 200;
    const TEMP_FILE: &str = "temp_raw_indicators.csv";
    const FINAL_FILE: &str = "tuning_output_head.csv";
    const HEAD_LIMIT: usize = 1000;

    macro_rules! log {
        ($icon:expr, $msg:expr) => {
            println!("[{:>8.2?}] {} {}", start_app.elapsed(), $icon, $msg);
        };
        ($icon:expr, $fmt:expr, $($arg:tt)*) => {
            println!("[{:>8.2?}] {} {}", start_app.elapsed(), $icon, format!($fmt, $($arg)*));
        };
    }

    log!("🚀", "Starting Application...");

    // 1. Load Config
    let config = AppConfig::load()?;

    // 2. Connect
    log!("🔌", "Connecting to ClickHouse at {}...", config.db_url);
    let repository = ClickHouseRepository::new(&config.db_url);
    let mut client = repository.get_connection().await?;

    // 3. Build Query
    let sql_query = ClickHouseRepository::build_candles_query(&config);

    log!("🌊", "Executing Streaming Query...");

    // --- STREAMING STATE ---
    let mut buf_opens: Vec<f64> = Vec::new();
    let mut buf_highs: Vec<f64> = Vec::new();
    let mut buf_lows: Vec<f64> = Vec::new();
    let mut buf_closes: Vec<f64> = Vec::new();
    let mut buf_volumes: Vec<f64> = Vec::new();

    let mut cleaner: Option<Phase1Cleaner> = None;
    let mut csv_writer: Option<arrow::csv::Writer<File>> = None;
    let mut saved_schema: Option<Arc<Schema>> = None;
    let mut total_rows_processed = 0;

    let mut stream = client.query(&sql_query).stream_blocks();

    while let Some(block) = stream.next().await {
        let loop_start = Instant::now();
        let block = block?;
        if block.rows().count() == 0 { continue; }

        let (mut columns, math_data, mut fields) = parse_block(&block)?;
        let row_count = math_data.closes.len();

        // Merge Buffer
        let full_opens = [buf_opens.as_slice(), math_data.opens.as_slice()].concat();
        let full_highs = [buf_highs.as_slice(), math_data.highs.as_slice()].concat();
        let full_lows = [buf_lows.as_slice(), math_data.lows.as_slice()].concat();
        let full_closes = [buf_closes.as_slice(), math_data.closes.as_slice()].concat();
        let full_volumes = [buf_volumes.as_slice(), math_data.volumes.as_slice()].concat();

        if full_closes.len() <= MAX_PERIOD {
            buf_opens = full_opens; buf_highs = full_highs; buf_lows = full_lows;
            buf_closes = full_closes; buf_volumes = full_volumes;
            log!("⏳", "Buffering initial data...");
            continue;
        }

        // Calculate Indicators
        let offset = buf_opens.len();
        indicators::append_indicators(
            &mut fields, &mut columns,
            &full_opens, &full_highs, &full_lows, &full_closes, &full_volumes,
            5, 200, offset
        );

        // Create Batch
        let schema = Schema::new(fields);
        if saved_schema.is_none() { saved_schema = Some(Arc::new(schema.clone())); }
        let batch = RecordBatch::try_new(Arc::new(schema), columns)?;
        if cleaner.is_none() { cleaner = Some(Phase1Cleaner::new(batch.schema())); }
        if let Some(c) = &mut cleaner { c.check_batch(&batch); }
        if csv_writer.is_none() {
            let file = File::create(TEMP_FILE)?;
            csv_writer = Some(WriterBuilder::new().with_header(true).build(file));
        }
        if let Some(w) = &mut csv_writer { w.write(&batch)?; }

        // Update Buffer
        buf_opens = get_tail(&full_opens, MAX_PERIOD);
        buf_highs = get_tail(&full_highs, MAX_PERIOD);
        buf_lows = get_tail(&full_lows, MAX_PERIOD);
        buf_closes = get_tail(&full_closes, MAX_PERIOD);
        buf_volumes = get_tail(&full_volumes, MAX_PERIOD);

        total_rows_processed += row_count;
        log!("📦", "Processed batch: {} rows. (Total: {}). Duration: {:.2?}", row_count, total_rows_processed, loop_start.elapsed());
    }

    drop(csv_writer);

    // --- 4. FINALIZE (EXPORT HEAD ONLY) ---
    log!("🧹", "Finalizing: Exporting HEAD SAMPLE ONLY...");

    if let Some(c) = cleaner {
        let (keep_cols, dropped) = c.get_results();
        log!("📊", "Phase 1 Analysis (Based on FULL DATA):");
        log!("   ", "- Kept Columns: {}", keep_cols.len());
        log!("   ", "- Dropped Columns: {}", dropped.len());

        let file = File::open(TEMP_FILE)?;
        let schema_ref = saved_schema.ok_or("No schema available")?;
        let mut reader = ReaderBuilder::new(schema_ref).with_header(true).build(file)?;
        let schema = reader.schema();
        let mut indices = Vec::new();
        for name in &keep_cols {
            if let Ok(idx) = schema.index_of(name) { indices.push(idx); }
        }

        let final_file = File::create(FINAL_FILE)?;
        let mut final_writer = WriterBuilder::new().with_header(true).build(final_file);

        let finalize_start = Instant::now();
        let mut rows_written = 0;
        while let Some(batch_res) = reader.next() {
            if rows_written >= HEAD_LIMIT {
                break;
            }
            let batch = batch_res?;
            let clean_rows = filter_null_rows(&batch)?;
            if clean_rows.num_rows() > 0 {
                let projected = clean_rows.project(&indices)?;
                let remaining = HEAD_LIMIT - rows_written;
                let rows_to_write = std::cmp::min(projected.num_rows(), remaining);
                let slice = projected.slice(0, rows_to_write);

                final_writer.write(&slice)?;
                rows_written += rows_to_write;
            }
        }

        fs::remove_file(TEMP_FILE)?;

        log!("💾", "Sample Export Complete: Written {} rows to '{}' in {:.2?}", rows_written, FINAL_FILE, finalize_start.elapsed());
        log!("ℹ️", "Note: Phase 1 analysis used ALL data, but CSV only contains the top {} rows for inspection.", rows_written);

    } else {
        log!("⚠️", "No data processed.");
    }

    log!("🎉", "All Done! Total Execution Time: {:.2?}", start_app.elapsed());
    Ok(())
}