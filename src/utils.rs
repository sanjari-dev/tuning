use arrow::record_batch::RecordBatch;
use arrow::compute::filter_record_batch;
use arrow::array::BooleanArray;
use std::error::Error;

pub fn filter_null_rows(batch: &RecordBatch) -> Result<RecordBatch, Box<dyn Error>> {
    let num_rows = batch.num_rows();
    if num_rows == 0 { return Ok(batch.clone()); }

    let mut keep_row = vec![true; num_rows];
    for col in batch.columns() {
        if col.null_count() > 0 {
            for i in 0..num_rows {
                if col.is_null(i) { keep_row[i] = false; }
            }
        }
    }
    let predicate = BooleanArray::from(keep_row);
    let filtered_batch = filter_record_batch(batch, &predicate)?;
    Ok(filtered_batch)
}

pub fn get_tail(data: &[f64], n: usize) -> Vec<f64> {
    if data.len() <= n {
        data.to_vec()
    } else {
        data[data.len() - n..].to_vec()
    }
}