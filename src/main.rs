mod config;
mod conversion;
mod indicators;
mod phase1;
mod phase2;
mod utils;
mod repository;

use arrow::array::{Array, RecordBatch, Float64Array};
use arrow::csv::WriterBuilder;
use arrow::datatypes::Schema;
use futures_util::stream::StreamExt;
use std::error::Error;
use std::fs::File;
use std::sync::Arc;
use std::time::Instant;
use ndarray::Array2;
use std::cmp;

use config::AppConfig;
use conversion::parse_block;
use phase1::Phase1Cleaner;
use phase2::Phase2Streaming;
use repository::ClickHouseRepository;
use utils::{filter_null_rows, get_tail};

const PROTECTED_COLS: &[&str] = &[
    "timestamp", "instrument", "timeframe",
    "open", "high", "low", "close",
    "tick_count", "min_spread", "max_spread", "avg_spread",
    "total_bid_volume", "total_ask_volume", "vwap",
    "time_minute_of_hour", "time_hour_of_day", "time_day_of_week"
];

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let start_app = Instant::now();
    const MAX_PERIOD: usize = 200;
    const FINAL_FILE: &str = "tuning_output_head.csv";
    const EXPORT_LIMIT: usize = 500;
    macro_rules! log {
        ($icon:expr, $msg:expr) => {
            println!("[{:>8.2?}] {} {}", start_app.elapsed(), $icon, $msg);
        };
        ($icon:expr, $fmt:expr, $($arg:tt)*) => {
            println!("[{:>8.2?}] {} {}", start_app.elapsed(), $icon, format!($fmt, $($arg)*));
        };
    }

    log!("🚀", "Application Startup: Initializing Feature Tuning Pipeline");

    // --- SCOPE: CONFIGURATION & CONNECTION ---
    let config = AppConfig::load()?;
    log!("⚙️", "Configuration Loaded: Instrument={}, Timeframe={}", config.query_instrument, config.query_timeframe);

    let repository = ClickHouseRepository::new(&config.db_url);
    log!("🔌", "Connecting to ClickHouse database");
    let mut client_analysis = repository.get_connection().await?;
    let sql_query = ClickHouseRepository::build_candles_query(&config);
    log!("📝", "SQL Query prepared successfully");

    // ==========================================
    // PASS 1: ANALYSIS STREAM
    // ==========================================
    log!("🌊", ">>> START PASS 1: Analysis Stream (Single-Pass Statistics) <<<");
    log!("ℹ️", "Processing Mode: Streaming. No intermediate temp files will be created");

    let mut buf_opens: Vec<f64> = Vec::new();
    let mut buf_highs: Vec<f64> = Vec::new();
    let mut buf_lows: Vec<f64> = Vec::new();
    let mut buf_closes: Vec<f64> = Vec::new();
    let mut buf_volumes: Vec<f64> = Vec::new();

    let mut cleaner: Option<Phase1Cleaner> = None;
    let mut phase2: Option<Phase2Streaming> = None;
    let mut phase2_col_indices: Vec<usize> = Vec::new();

    let mut total_rows = 0;

    // SCOPE: Analysis Loop
    {
        log!("🔄", "Executing Query and starting data stream loop");
        let mut stream = client_analysis.query(&sql_query).stream_blocks();

        while let Some(block) = stream.next().await {
            let block = block?;
            if block.rows().count() == 0 { continue; }

            // SCOPE: Parsing & Buffering
            let (mut columns, math_data, mut fields) = parse_block(&block)?;
            let row_count = math_data.closes.len();

            let full_opens = [buf_opens.as_slice(), math_data.opens.as_slice()].concat();
            let full_highs = [buf_highs.as_slice(), math_data.highs.as_slice()].concat();
            let full_lows = [buf_lows.as_slice(), math_data.lows.as_slice()].concat();
            let full_closes = [buf_closes.as_slice(), math_data.closes.as_slice()].concat();
            let full_volumes = [buf_volumes.as_slice(), math_data.volumes.as_slice()].concat();

            // Check Buffer Warmup
            if full_closes.len() <= MAX_PERIOD {
                // log!("⏳", "Buffering initial data (Current: {} rows)", full_closes.len());
                buf_opens = full_opens; buf_highs = full_highs; buf_lows = full_lows;
                buf_closes = full_closes; buf_volumes = full_volumes;
                continue;
            }

            // SCOPE: Indicator Calculation
            let offset = buf_opens.len();
            indicators::append_indicators(
                &mut fields, &mut columns,
                &full_opens, &full_highs, &full_lows, &full_closes, &full_volumes,
                5, 200, offset
            );

            let schema = Arc::new(Schema::new(fields));
            let batch = RecordBatch::try_new(schema.clone(), columns)?;

            // --- SCOPE: PHASE 1 (Variance Check) ---
            if cleaner.is_none() {
                log!("🧹", "Phase 1 Initialized: Variance Cleaner started tracking {} columns", batch.num_columns());
                cleaner = Some(Phase1Cleaner::new(batch.schema()));
            }
            if let Some(c) = &mut cleaner { c.check_batch(&batch); }

            // --- SCOPE: PHASE 2 (Matrix Accumulation) ---
            let clean_batch = filter_null_rows(&batch)?;
            if clean_batch.num_rows() > 0 {
                let num_rows = clean_batch.num_rows();

                // Initialization Scope
                if phase2.is_none() {
                    log!("🔬", "Phase 2 Initialized: Identifying candidate features for Correlation Analysis");
                    let mut candidates = Vec::new();
                    let mut indices = Vec::new();

                    let schema_ref = clean_batch.schema();
                    for (i, field) in schema_ref.fields().iter().enumerate() {
                        let name = field.name();
                        let is_float = field.data_type() == &arrow::datatypes::DataType::Float64;
                        let is_protected = PROTECTED_COLS.contains(&name.as_str()) ||
                            name.starts_with("target_") ||
                            name.starts_with("atr_");

                        if is_float && !is_protected {
                            candidates.push(name.clone());
                            indices.push(i);
                        }
                    }
                    log!("📋", "Phase 2 Candidates Selected: {} features tracking", candidates.len());
                    phase2 = Some(Phase2Streaming::new(candidates));
                    phase2_col_indices = indices;
                }

                // Accumulation Scope
                if let Some(p2) = &mut phase2 {
                    let num_features = phase2_col_indices.len();
                    let mut batch_vec = Vec::with_capacity(num_rows * num_features);

                    for r in 0..num_rows {
                        for &col_idx in &phase2_col_indices {
                            let col = clean_batch.column(col_idx);
                            let val = col.as_any().downcast_ref::<Float64Array>().unwrap().value(r);
                            batch_vec.push(val);
                        }
                    }

                    let matrix = Array2::from_shape_vec((num_rows, num_features), batch_vec)?;
                    p2.add_batch(&matrix);
                }
            }

            // SCOPE: Buffer Cleanup
            buf_opens = get_tail(&full_opens, MAX_PERIOD);
            buf_highs = get_tail(&full_highs, MAX_PERIOD);
            buf_lows = get_tail(&full_lows, MAX_PERIOD);
            buf_closes = get_tail(&full_closes, MAX_PERIOD);
            buf_volumes = get_tail(&full_volumes, MAX_PERIOD);

            total_rows += row_count;
            if total_rows % 50000 == 0 {
                log!("⏱️", "[Pass 1 Progress] Analyzed {} rows so far", total_rows);
            }
        }
    } // End Analysis Loop
    log!("✅", "Pass 1 Analysis Loop Completed. Total rows analyzed: {}", total_rows);

    // ==========================================
    // SCOPE: RESULTS COMPUTATION
    // ==========================================
    log!("🧠", ">>> START COMPUTATION: Finalizing Feature Selection <<<");
    let mut final_selected_cols = Vec::new();

    if let Some(c) = cleaner {
        let (keep_cols_p1, dropped_p1) = c.get_results();
        log!("📊", "[Phase 1 Result] Variance Check Complete");
        log!("   ", "-> Kept: {} features", keep_cols_p1.len());
        log!("   ", "-> Dropped: {} features (Zero Variance)", dropped_p1.len());

        if let Some(p2) = phase2 {
            log!("🧮", "[Phase 2 Computation] Constructing Correlation Matrix from accumulators");

            // Finalize calculation
            let kept_p2 = p2.finalize_and_cluster(&keep_cols_p1, 0.95);

            log!("🧬", "[Phase 2 Result] Clustering Complete");
            log!("   ", "-> Input Candidates: {}", p2.feature_names.len());
            log!("   ", "-> Selected Representatives: {}", kept_p2.len());
            log!("   ", "-> Redundant Removed: {}", p2.feature_names.len() - kept_p2.len());

            // Merging Columns
            let protected_kept: Vec<String> = keep_cols_p1.iter()
                .filter(|&n| PROTECTED_COLS.contains(&n.as_str()) || n.starts_with("atr_"))
                .cloned()
                .collect();

            log!("🔗", "Merging Protected Columns ({}) with Phase 2 Selection ({})", protected_kept.len(), kept_p2.len());
            final_selected_cols.extend(protected_kept);
            final_selected_cols.extend(kept_p2);
            final_selected_cols.sort();
            final_selected_cols.dedup();
        }
    } else {
        log!("⚠️", "Analysis aborted: No data was processed during Pass 1");
        return Ok(());
    }

    log!("✅", "Final Feature Selection Set: {} unique columns determined", final_selected_cols.len());

    // ==========================================
    // PASS 2: EXPORT STREAM (HEAD 500)
    // ==========================================
    log!("💾", ">>> START PASS 2: Data Export Stream (Head Sample) <<<");
    log!("ℹ️", "Re-streaming data to export TOP {} rows to '{}' for inspection", EXPORT_LIMIT, FINAL_FILE);

    // New Connection Scope
    let mut client_export = repository.get_connection().await?;
    let mut stream_export = client_export.query(&sql_query).stream_blocks();

    let file_out = File::create(FINAL_FILE)?;
    let mut csv_writer = WriterBuilder::new().with_header(true).build(file_out);

    // Reset Buffers for Pass 2
    let mut buf_opens = Vec::new();
    let mut buf_highs = Vec::new();
    let mut buf_lows = Vec::new();
    let mut buf_closes = Vec::new();
    let mut buf_volumes = Vec::new();
    let mut output_indices: Option<Vec<usize>> = None;
    let mut exported_rows = 0;

    while let Some(block) = stream_export.next().await {
        if exported_rows >= EXPORT_LIMIT {
            break;
        }

        let block = block?;
        if block.rows().count() == 0 { continue; }

        let (mut columns, math_data, mut fields) = parse_block(&block)?;

        let full_opens = [buf_opens.as_slice(), math_data.opens.as_slice()].concat();
        let full_highs = [buf_highs.as_slice(), math_data.highs.as_slice()].concat();
        let full_lows = [buf_lows.as_slice(), math_data.lows.as_slice()].concat();
        let full_closes = [buf_closes.as_slice(), math_data.closes.as_slice()].concat();
        let full_volumes = [buf_volumes.as_slice(), math_data.volumes.as_slice()].concat();

        if full_closes.len() <= MAX_PERIOD {
            buf_opens = full_opens; buf_highs = full_highs; buf_lows = full_lows;
            buf_closes = full_closes; buf_volumes = full_volumes;
            continue;
        }

        let offset = buf_opens.len();
        indicators::append_indicators(
            &mut fields, &mut columns,
            &full_opens, &full_highs, &full_lows, &full_closes, &full_volumes,
            5, 200, offset
        );

        let schema = Schema::new(fields);
        let batch = RecordBatch::try_new(Arc::new(schema.clone()), columns)?;
        let clean_batch = filter_null_rows(&batch)?;

        if clean_batch.num_rows() > 0 {
            // Index Mapping Scope (Once)
            if output_indices.is_none() {
                log!("🗺️", "Mapping final column indices for export");
                let mut idxs = Vec::new();
                for name in &final_selected_cols {
                    if let Ok(i) = schema.index_of(name) {
                        idxs.push(i);
                    }
                }
                log!("📝", "Column mapping complete. Writing CSV header and data");
                output_indices = Some(idxs);
            }

            // Writing Scope (Slicing logic)
            if let Some(idxs) = &output_indices {
                let remaining = EXPORT_LIMIT - exported_rows;
                let rows_to_take = cmp::min(remaining, clean_batch.num_rows());
                let slice = clean_batch.slice(0, rows_to_take);
                let projected = slice.project(idxs)?;

                csv_writer.write(&projected)?;
                exported_rows += rows_to_take;
            }
        }

        buf_opens = get_tail(&full_opens, MAX_PERIOD);
        buf_highs = get_tail(&full_highs, MAX_PERIOD);
        buf_lows = get_tail(&full_lows, MAX_PERIOD);
        buf_closes = get_tail(&full_closes, MAX_PERIOD);
        buf_volumes = get_tail(&full_volumes, MAX_PERIOD);
    }

    log!("✅", "Sample Export Completed. Total rows written: {}", exported_rows);
    log!("🎉", "Pipeline Execution Successfully Completed!");
    log!("⏱️", "Total Runtime: {:.2?}", start_app.elapsed());
    Ok(())
}