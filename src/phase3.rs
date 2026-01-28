use std::collections::HashMap;
use std::time::Instant;

pub struct Phase3Accumulator {
    batch_history: HashMap<String, Vec<f64>>,
    pub feature_names: Vec<String>,
    pub batch_count: usize,
}

impl Phase3Accumulator {
    pub fn new(feature_names: Vec<String>) -> Self {
        let mut batch_history = HashMap::new();
        for name in &feature_names {
            batch_history.insert(name.clone(), Vec::new());
        }

        Self {
            batch_history,
            feature_names,
            batch_count: 0,
        }
    }

    pub fn process_batch(&mut self, features: &Vec<Vec<f64>>, targets: &Vec<f64>) {
        if features.is_empty() || features.len() != targets.len() {
            return;
        }

        self.batch_count += 1;

        let mut local_stats: Vec<(f64, f64, f64, f64, f64, f64)> =
            vec![(0.0, 0.0, 0.0, 0.0, 0.0, 0.0); self.feature_names.len()];

        for (row_idx, row_data) in features.iter().enumerate() {
            let y = targets[row_idx];
            if !y.is_finite() {
                continue;
            }

            for (col_idx, &x) in row_data.iter().enumerate() {
                if !x.is_finite() {
                    continue;
                }

                let stat = &mut local_stats[col_idx];
                stat.0 += 1.0;
                stat.1 += x;
                stat.2 += y;
                stat.3 += x * x;
                stat.4 += y * y;
                stat.5 += x * y;
            }
        }

        for (col_idx, (n, sum_x, sum_y, sum_xx, sum_yy, sum_xy)) in
            local_stats.into_iter().enumerate()
        {
            if n < 2.0 {
                if let Some(name) = self.feature_names.get(col_idx) {
                    if let Some(history) = self.batch_history.get_mut(name) {
                        history.push(0.0);
                    }
                }
                continue;
            }

            let numerator = n * sum_xy - sum_x * sum_y;
            let var_x = n * sum_xx - sum_x * sum_x;
            let var_y = n * sum_yy - sum_y * sum_y;
            let denominator = (var_x * var_y).sqrt();

            let r = if denominator > 1e-9 {
                numerator / denominator
            } else {
                0.0
            };

            if let Some(name) = self.feature_names.get(col_idx) {
                if let Some(history) = self.batch_history.get_mut(name) {
                    history.push(r.abs());
                }
            }
        }
    }

    pub fn finalize_selection(
        &self,
        candidates_from_phase2: &[String],
        top_n: usize,
        start_time: Instant,
    ) -> Vec<String> {
        let log = |icon: &str, msg: String| {
            println!("[{:>8.2?}] {} {}", start_time.elapsed(), icon, msg);
        };

        log("🏆", "Calculating Statistics (Phase 3 & 4)...".to_string());

        struct FeatureStats<'a> {
            name: &'a String,
            mean: f64,
            std_dev: f64,
            stability_score: f64,
        }

        let mut all_stats: Vec<FeatureStats> = Vec::new();
        let stability_penalty = 1.0;

        for name in candidates_from_phase2 {
            if let Some(history) = self.batch_history.get(name) {
                if history.is_empty() {
                    continue;
                }

                let sum: f64 = history.iter().sum();
                let mean = sum / history.len() as f64;
                let variance: f64 = history
                    .iter()
                    .map(|value| {
                        let diff = mean - value;
                        diff * diff
                    })
                    .sum::<f64>()
                    / history.len() as f64;
                let std_dev = variance.sqrt();

                let stability_score = mean - (stability_penalty * std_dev);

                all_stats.push(FeatureStats {
                    name,
                    mean,
                    std_dev,
                    stability_score,
                });
            }
        }

        all_stats.sort_by(|a, b| {
            b.mean
                .partial_cmp(&a.mean)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        println!(
            "\n📊 [Phase 3 Result] Top Features by Pure Avg Correlation (Before Stability Penalty):"
        );
        println!(
            "{:<50} | {:<10} | {:<10}",
            "Feature Name", "Avg Corr", "Volatility"
        );
        println!("{:-<76}", "");

        for (i, stats) in all_stats.iter().take(15).enumerate() {
            println!(
                "{:<50} | {:<10.4} | {:<10.4}",
                format!("{}. {}", i + 1, stats.name),
                stats.mean,
                stats.std_dev
            );
        }
        println!();

        all_stats.sort_by(|a, b| {
            b.stability_score
                .partial_cmp(&a.stability_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        log(
            "⚖️",
            format!(
                "Applying Phase 4 Penalty: Score = Mean - ({:.1} * StdDev)",
                stability_penalty
            ),
        );
        println!("\n🚀 [Phase 4 Result] Final Ranking by Stability Score:");
        println!(
            "{:<50} | {:<10} | {:<10} | {:<10}",
            "Feature Name", "FinalScore", "Avg Corr", "Volatility"
        );
        println!("{:-<90}", "");

        for (i, stats) in all_stats.iter().take(20).enumerate() {
            println!(
                "{:<50} | {:<10.4} | {:<10.4} | {:<10.4}",
                format!("{}. {}", i + 1, stats.name),
                stats.stability_score,
                stats.mean,
                stats.std_dev
            );
        }
        println!();

        all_stats
            .into_iter()
            .take(top_n)
            .map(|stats| stats.name.clone())
            .collect()
    }
}
