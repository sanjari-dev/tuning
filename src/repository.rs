use clickhouse_rs::{Pool, ClientHandle};
use std::error::Error;
use crate::config::AppConfig;

pub struct ClickHouseRepository {
    pool: Pool,
}

impl ClickHouseRepository {
    pub fn new(db_url: &str) -> Self {
        Self {
            pool: Pool::new(db_url),
        }
    }
    pub async fn get_connection(&self) -> Result<ClientHandle, Box<dyn Error>> {
        let client = self.pool.get_handle().await?;
        Ok(client)
    }
    pub fn build_candles_query(config: &AppConfig) -> String {
        format!(r#"
            SELECT
                toUnixTimestamp(timestamp) as ts,
                instrument as inst,
                timeframe as tf,
                toFloat64(open) as open,
                toFloat64(high) as high,
                toFloat64(low) as low,
                toFloat64(close) as close,
                tick_count,
                toFloat64(min_spread) as min_spr,
                toFloat64(max_spread) as max_spr,
                toFloat64(avg_spread) as avg_spr,
                total_bid_volume,
                total_ask_volume,
                toFloat64(vwap) as vwap
            FROM candles
            WHERE
                instrument = '{}'
                AND timeframe = '{}'
                AND timestamp >= toDateTime('{}')
                AND timestamp <= toDateTime('{}')
            ORDER BY timestamp ASC
        "#,
                config.query_instrument,
                config.query_timeframe,
                config.query_start,
                config.query_end)
    }
}