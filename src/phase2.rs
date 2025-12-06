use ndarray::{Array1, Array2, Axis};
use kodama::{linkage, Method};
use std::collections::{HashMap, HashSet};

macro_rules! log {
    ($icon:expr, $msg:expr) => {
        println!("{} [Phase 2] {}", $icon, $msg);
    };
    ($icon:expr, $fmt:expr, $($arg:tt)*) => {
        println!("{} [Phase 2] {}", $icon, format!($fmt, $($arg)*));
    };
}

pub struct Phase2Streaming {
    pub feature_names: Vec<String>,
    sum_x: Array1<f64>,
    sum_xy: Array2<f64>,
    n: usize,
}

impl Phase2Streaming {
    pub fn new(feature_names: Vec<String>) -> Self {
        let num_features = feature_names.len();
        log!("🔬", "Initialized tracking {} candidate features", num_features);
        Self {
            feature_names,
            sum_x: Array1::zeros(num_features),
            sum_xy: Array2::zeros((num_features, num_features)),
            n: 0,
        }
    }

    pub fn add_batch(&mut self, batch_matrix: &Array2<f64>) {
        let batch_rows = batch_matrix.nrows();
        if batch_rows == 0 { return; }

        self.n += batch_rows;

        let batch_sum = batch_matrix.sum_axis(Axis(0));
        self.sum_x = &self.sum_x + &batch_sum;

        let batch_cooccurrence = batch_matrix.t().dot(batch_matrix);
        self.sum_xy = &self.sum_xy + &batch_cooccurrence;
    }

    pub fn finalize_and_cluster(&self, valid_features_p1: &[String], threshold: f64) -> Vec<String> {
        let n = self.n as f64;
        let valid_set: HashSet<&String> = valid_features_p1.iter().collect();
        let mut valid_indices = Vec::new();
        let mut final_names = Vec::new();

        for (i, name) in self.feature_names.iter().enumerate() {
            if valid_set.contains(name) {
                valid_indices.push(i);
                final_names.push(name.clone());
            }
        }

        let num_valid = valid_indices.len();
        log!("🧠", "Finalizing: Input {} features -> Valid Phase 1 {} features", self.feature_names.len(), num_valid);

        if num_valid < 2 {
            log!("⚠️", "Not enough features to cluster. Returning all valid features");
            return final_names;
        }

        log!("🧮", "Reconstructing Correlation Matrix from accumulators");

        let mut condensed_dist = Vec::with_capacity(num_valid * (num_valid - 1) / 2);
        let mut means = Vec::with_capacity(num_valid);
        let mut std_devs = Vec::with_capacity(num_valid);

        for &idx in &valid_indices {
            let sum = self.sum_x[idx];
            let sum_sq = self.sum_xy[[idx, idx]];
            let mean = sum / n;
            let variance = (sum_sq - n * mean * mean) / (n - 1.0);

            means.push(mean);
            std_devs.push(variance.max(0.0).sqrt());
        }

        for i in 0..num_valid {
            for j in (i + 1)..num_valid {
                let real_i = valid_indices[i];
                let real_j = valid_indices[j];

                let sum_xy = self.sum_xy[[real_i, real_j]];
                let covariance = (sum_xy - n * means[i] * means[j]) / (n - 1.0);
                let denom = std_devs[i] * std_devs[j];

                let corr = if denom.abs() < 1e-9 {
                    0.0
                } else {
                    (covariance / denom).clamp(-1.0, 1.0)
                };

                let dist = (1.0 - corr.abs()).max(0.0);
                condensed_dist.push(dist);
            }
        }

        log!("🧬", "Running Hierarchical Clustering (Linkage)");

        let dendrogram = linkage(&mut condensed_dist, num_valid, Method::Average);
        let dist_threshold = 1.0 - threshold;

        let mut clusters: HashMap<usize, Vec<usize>> = HashMap::new();
        for i in 0..num_valid {
            clusters.insert(i, vec![i]);
        }

        let mut next_cluster_id = num_valid;
        for step in dendrogram.steps() {
            if step.dissimilarity > dist_threshold { break; }

            let c1 = step.cluster1;
            let c2 = step.cluster2;

            if let (Some(m1), Some(m2)) = (clusters.remove(&c1), clusters.remove(&c2)) {
                let mut new_m = m1;
                new_m.extend(m2);
                clusters.insert(next_cluster_id, new_m);
            }
            next_cluster_id += 1;
        }

        let mut kept_names = Vec::new();
        let mut kept_indices_local = HashSet::new();

        for (_, mut members) in clusters {
            if !members.is_empty() {
                members.sort();
                kept_indices_local.insert(members[0]);
            }
        }

        for i in 0..num_valid {
            if kept_indices_local.contains(&i) {
                kept_names.push(final_names[i].clone());
            }
        }

        log!("✅", "Clustering complete. Kept {} features out of {}", kept_names.len(), num_valid);
        kept_names
    }
}