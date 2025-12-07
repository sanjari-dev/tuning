mod config;
mod conversion;
mod indicators;
mod phase1;
mod phase2;
mod utils;
mod repository;
mod pipeline;
mod gpu_indicators;

use config::AppConfig;
use pipeline::TuningPipeline;
use std::error::Error;
use std::time::Instant;

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let start_app = Instant::now();
    macro_rules! log {
        ($msg:expr) => {
            println!("[{:>8.2?}] {}", start_app.elapsed(), $msg);
        };
        ($fmt:expr, $($arg:tt)*) => {
            println!("[{:>8.2?}] {}", start_app.elapsed(), format!($fmt, $($arg)*));
        };
    }

    log!("🚀 Application Startup...");

    let config = AppConfig::load()?;
    log!("[Init] Configuration Loaded: {} ({})", config.query_instrument, config.query_timeframe);

    let pipeline = TuningPipeline::new(config, start_app);

    let final_features = pipeline.run_analysis().await?;
    log!("[Result] Final Feature Set: {} features selected.", final_features.len());

    if final_features.is_empty() {
        log!("⚠️ No features selected (or analysis failed). Exiting.");
        return Ok(());
    }

    pipeline.run_export(&final_features).await?;

    log!("🎉 All Done! Total Runtime: {:.2?}", start_app.elapsed());
    Ok(())
}