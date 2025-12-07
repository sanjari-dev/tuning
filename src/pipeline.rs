use std::error::Error;
use std::fs::File;
use std::sync::Arc;
use std::cmp;
use std::time::Instant;
use futures_util::stream::StreamExt;
use arrow::array::{Array, RecordBatch, Float64Array};
use arrow::datatypes::Schema;
use ndarray::Array2;
use parquet::arrow::arrow_writer::ArrowWriter;
use parquet::file::properties::WriterProperties;

use crate::config::AppConfig;
use crate::repository::ClickHouseRepository;
use crate::phase1::Phase1Cleaner;
use crate::phase2::Phase2Streaming;
use crate::conversion::parse_block;
use crate::gpu_indicators::GpuIndicatorHelper;
use crate::indicators;
use crate::utils::{filter_null_rows, get_tail};

const PROTECTED_COLS: &[&str] = &[
    "timestamp", "instrument", "timeframe",
    "open", "high", "low", "close",
    "tick_count", "min_spread", "max_spread", "avg_spread",
    "total_bid_volume", "total_ask_volume", "vwap",
    "time_minute_of_hour", "time_hour_of_day", "time_day_of_week"
];

const MAX_PERIOD: usize = 200;
const EXPORT_LIMIT: usize = 500;
const FINAL_FILE: &str = "tuning_output_head.parquet";

pub struct TuningPipeline {
    config: AppConfig,
    repository: ClickHouseRepository,
    start_time: Instant,
}

impl TuningPipeline {
    pub fn new(config: AppConfig, start_time: Instant) -> Self {
        let repository = ClickHouseRepository::new(&config.db_url);
        Self { config, repository, start_time }
    }

    fn log(&self, icon: &str, msg: impl std::fmt::Display) {
        println!("[{:>8.2?}] {} {}", self.start_time.elapsed(), icon, msg);
    }

    pub async fn run_analysis(&self) -> Result<Vec<String>, Box<dyn Error>> {
        self.log("🌊", ">>> START PASS 1: Analysis Stream (Single-Pass Statistics) <<<");

        let sql = ClickHouseRepository::build_candles_query(&self.config);
        let mut client = self.repository.get_connection().await?;
        let mut stream = client.query(&sql).stream_blocks();
        let mut buf_opens: Vec<f64> = Vec::new();
        let mut buf_highs: Vec<f64> = Vec::new();
        let mut buf_lows: Vec<f64> = Vec::new();
        let mut buf_closes: Vec<f64> = Vec::new();
        let mut buf_volumes: Vec<f64> = Vec::new();
        let mut cleaner: Option<Phase1Cleaner> = None;
        let mut phase2: Option<Phase2Streaming> = None;
        let mut phase2_col_indices: Vec<usize> = Vec::new();
        let mut full_schema_names: Vec<String> = Vec::new();
        let mut total_rows = 0;

        while let Some(block) = stream.next().await {
            let block = block?;
            if block.rows().count() == 0 { continue; }

            let (mut columns, math_data, mut fields) = parse_block(&block)?;
            let row_count = math_data.closes.len();
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
            let gpu_helper = GpuIndicatorHelper::new(&full_closes, &full_highs, &full_lows);
            indicators::append_indicators(
                &mut fields, &mut columns,
                &full_opens, &full_highs, &full_lows, &full_closes, &full_volumes,
                5, 200, offset,
                Some(&gpu_helper)
            );

            let schema = Arc::new(Schema::new(fields));
            let batch = RecordBatch::try_new(schema.clone(), columns)?;

            if full_schema_names.is_empty() {
                full_schema_names = schema.fields().iter().map(|f| f.name().clone()).collect();
            }

            if cleaner.is_none() {
                self.log("🧹", format!("Phase 1 Initialized: Tracking {} columns", batch.num_columns()));
                cleaner = Some(Phase1Cleaner::new(batch.schema(), self.start_time));
            }
            if let Some(c) = &mut cleaner { c.check_batch(&batch); }

            let clean_batch = filter_null_rows(&batch)?;
            if clean_batch.num_rows() > 0 {
                let num_rows = clean_batch.num_rows();

                if phase2.is_none() {
                    self.log("🔬", "Phase 2 Initialized: Selecting candidates");
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
                    self.log("📋", format!("Phase 2 Candidates: {}", candidates.len()));
                    phase2 = Some(Phase2Streaming::new(candidates, self.start_time));
                    phase2_col_indices = indices;
                }

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

            buf_opens = get_tail(&full_opens, MAX_PERIOD);
            buf_highs = get_tail(&full_highs, MAX_PERIOD);
            buf_lows = get_tail(&full_lows, MAX_PERIOD);
            buf_closes = get_tail(&full_closes, MAX_PERIOD);
            buf_volumes = get_tail(&full_volumes, MAX_PERIOD);

            total_rows += row_count;
            if total_rows % 50000 == 0 {
                self.log("⏱️", format!("[Pass 1] Processed {} rows...", total_rows));
            }
        }

        self.log("✅", format!("Pass 1 Analysis Loop Completed. Total rows analyzed: {}", total_rows));

        let mut final_cols = Vec::new();
        if let Some(c) = cleaner {
            let (keep_p1, _dropped) = c.get_results();

            if let Some(p2) = phase2 {
                let kept_p2 = p2.finalize_and_cluster(&keep_p1, 0.95);
                let protected_forced: Vec<String> = full_schema_names.iter()
                    .filter(|&n| PROTECTED_COLS.contains(&n.as_str()) || n.starts_with("atr_") || n.starts_with("target_"))
                    .cloned()
                    .collect();

                self.log("🔗", format!("Merging: Protected ({}) + Selected Phase 2 ({})", protected_forced.len(), kept_p2.len()));
                final_cols.extend(protected_forced);
                final_cols.extend(kept_p2);
                final_cols.sort();
                final_cols.dedup();
            }
        }

        Ok(final_cols)
    }

    pub async fn run_export(&self, selected_cols: &[String]) -> Result<(), Box<dyn Error>> {
        self.log("💾", ">>> START PASS 2: Export to Parquet <<<");
        self.log("ℹ️", format!("Exporting top {} rows to '{}'", EXPORT_LIMIT, FINAL_FILE));

        let file_out = File::create(FINAL_FILE)?;
        let sql = ClickHouseRepository::build_candles_query(&self.config);
        let mut client = self.repository.get_connection().await?;
        let mut stream = client.query(&sql).stream_blocks();
        let mut parquet_writer: Option<ArrowWriter<File>> = None;
        let mut buf_opens = Vec::new();
        let mut buf_highs = Vec::new();
        let mut buf_lows = Vec::new();
        let mut buf_closes = Vec::new();
        let mut buf_volumes = Vec::new();
        let mut output_indices: Option<Vec<usize>> = None;
        let mut exported_rows = 0;

        while let Some(block) = stream.next().await {
            if exported_rows >= EXPORT_LIMIT { break; }
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
            let gpu_helper = GpuIndicatorHelper::new(&full_closes, &full_highs, &full_lows);
            indicators::append_indicators(
                &mut fields, &mut columns,
                &full_opens, &full_highs, &full_lows, &full_closes, &full_volumes,
                5, 200, offset,
                Some(&gpu_helper)
            );

            let schema = Arc::new(Schema::new(fields));
            let batch = RecordBatch::try_new(schema.clone(), columns)?;
            let clean_batch = filter_null_rows(&batch)?;

            if clean_batch.num_rows() > 0 {
                if output_indices.is_none() {
                    let mut idxs = Vec::new();
                    for name in selected_cols {
                        if let Ok(i) = schema.index_of(name) {
                            idxs.push(i);
                        }
                    }
                    let projected_schema = Arc::new(schema.project(&idxs)?);
                    let props = WriterProperties::builder().build();
                    parquet_writer = Some(ArrowWriter::try_new(file_out.try_clone()?, projected_schema, Some(props))?);
                    output_indices = Some(idxs);
                }

                if let Some(idxs) = &output_indices {
                    if let Some(writer) = &mut parquet_writer {
                        let remaining = EXPORT_LIMIT - exported_rows;
                        let rows_to_take = cmp::min(remaining, clean_batch.num_rows());
                        let slice = clean_batch.slice(0, rows_to_take);
                        let projected_batch = slice.project(idxs)?;
                        writer.write(&projected_batch)?;
                        exported_rows += rows_to_take;
                    }
                }
            }
            buf_opens = get_tail(&full_opens, MAX_PERIOD);
            buf_highs = get_tail(&full_highs, MAX_PERIOD);
            buf_lows = get_tail(&full_lows, MAX_PERIOD);
            buf_closes = get_tail(&full_closes, MAX_PERIOD);
            buf_volumes = get_tail(&full_volumes, MAX_PERIOD);
        }

        if let Some(writer) = parquet_writer {
            writer.close()?;
        }
        self.log("✅", format!("Export Completed. Total rows written: {}", exported_rows));
        Ok(())
    }
}