use arrow::array::{Array, Float64Array, RecordBatch};
use arrow::datatypes::{DataType, SchemaRef};
use std::collections::HashMap;

macro_rules! log {
    ($icon:expr, $msg:expr) => {
        println!("{} [Phase 1] {}", $icon, $msg);
    };
    ($icon:expr, $fmt:expr, $($arg:tt)*) => {
        println!("{} [Phase 1] {}", $icon, format!($fmt, $($arg)*));
    };
}

pub struct Phase1Cleaner {
    min_max: Vec<Option<(f64, f64)>>,
    duplicate_groups: Vec<Vec<usize>>,
    schema: SchemaRef,
    batch_counter: usize,
}

impl Phase1Cleaner {
    pub fn new(schema: SchemaRef) -> Self {
        let num_cols = schema.fields().len();
        let initial_group: Vec<usize> = (0..num_cols).collect();

        log!("🧹", "Initialized cleaner structure for {} columns", num_cols);

        Phase1Cleaner {
            min_max: vec![None; num_cols],
            duplicate_groups: vec![initial_group],
            schema,
            batch_counter: 0,
        }
    }

    pub fn check_batch(&mut self, batch: &RecordBatch) {
        self.batch_counter += 1;
        let num_cols = batch.num_columns();

        log!("🔍", "Processing Batch #{}: Scanning {} columns for variance & duplicates", self.batch_counter, num_cols);

        // --- 1. Min/Max Check for Zero Variance ---
        for i in 0..num_cols {
            let col = batch.column(i);
            if col.data_type() == &DataType::Float64 {
                if let Some(vals) = col.as_any().downcast_ref::<Float64Array>() {
                    let mut b_min = f64::INFINITY;
                    let mut b_max = f64::NEG_INFINITY;
                    let mut has_val = false;

                    for v in vals.iter().flatten() {
                        if v < b_min { b_min = v; }
                        if v > b_max { b_max = v; }
                        has_val = true;
                    }

                    if !has_val { continue; }
                    match self.min_max[i] {
                        None => {
                            self.min_max[i] = Some((b_min, b_max));
                        }
                        Some((curr_min, curr_max)) => {
                            self.min_max[i] = Some((curr_min.min(b_min), curr_max.max(b_max)));
                        }
                    }
                }
            }
        }

        // --- 2. Duplicate Detection ---
        let initial_groups_count = self.duplicate_groups.len();
        let mut new_groups: Vec<Vec<usize>> = Vec::new();

        for group in &self.duplicate_groups {
            if group.len() <= 1 {
                new_groups.push(group.clone());
                continue;
            }

            let mut sub_groups: Vec<Vec<usize>> = Vec::new();
            for &col_idx in group {
                let col_data = batch.column(col_idx);
                let mut found = false;

                for sub_group in &mut sub_groups {
                    let ref_idx = sub_group[0];
                    let ref_data = batch.column(ref_idx);
                    if col_data == ref_data {
                        sub_group.push(col_idx);
                        found = true;
                        break;
                    }
                }

                if !found {
                    sub_groups.push(vec![col_idx]);
                }
            }
            new_groups.extend(sub_groups);
        }

        if new_groups.len() > initial_groups_count {
            log!("⚡", "Duplicate groups refined: {} groups -> {} unique groups found in this batch", initial_groups_count, new_groups.len());
        }

        self.duplicate_groups = new_groups;
    }

    pub fn get_results(&self) -> (Vec<String>, Vec<String>) {
        log!("🧠", "Finalizing results. Analyzing variance statistics and duplicate groups");

        let mut keep_indices = Vec::new();
        let mut drop_reasons: HashMap<String, String> = HashMap::new();

        let fields = self.schema.fields();
        let mut zero_variance_indices = Vec::new();

        // 1. Identify Zero Variance
        for i in 0..fields.len() {
            let dtype = fields[i].data_type();
            if dtype == &DataType::Float64 {
                if let Some((min, max)) = self.min_max[i] {
                    if (max - min).abs() < 1e-9 {
                        zero_variance_indices.push(i);
                        drop_reasons.insert(fields[i].name().clone(), "Zero Variance (Constant Value)".to_string());
                    }
                } else {
                    zero_variance_indices.push(i);
                    drop_reasons.insert(fields[i].name().clone(), "All Null / Empty".to_string());
                }
            }
        }
        log!("📉", "Found {} features with Zero Variance or All-Nulls", zero_variance_indices.len());

        // 2. Resolve Duplicates
        let mut duplicate_drop_count = 0;
        for group in &self.duplicate_groups {
            if group.is_empty() { continue; }

            let mut sorted_group = group.clone();
            sorted_group.sort();

            let mut survivor = None;
            for &idx in &sorted_group {
                if !zero_variance_indices.contains(&idx) {
                    if survivor.is_none() {
                        survivor = Some(idx);
                        keep_indices.push(idx);
                    } else {
                        drop_reasons.insert(
                            fields[idx].name().clone(),
                            format!("Duplicate of {}", fields[survivor.unwrap()].name())
                        );
                        duplicate_drop_count += 1;
                    }
                }
            }
        }
        log!("👯", "Identified {} redundant features (Duplicates)", duplicate_drop_count);

        keep_indices.sort();

        let keep_names: Vec<String> = keep_indices.iter().map(|&i| fields[i].name().clone()).collect();
        let mut dropped_names: Vec<String> = Vec::new();

        for i in 0..fields.len() {
            if !keep_indices.contains(&i) {
                let name = fields[i].name();
                let reason = drop_reasons.get(name).map(|s| s.as_str()).unwrap_or("Unknown");
                dropped_names.push(format!("{} ({})", name, reason));
            }
        }

        log!("✅", "Result generation complete. Returning filter lists");
        (keep_names, dropped_names)
    }
}