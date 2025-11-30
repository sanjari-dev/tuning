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
        "#,
                config.query_instrument,
                config.query_timeframe,
                config.query_start,
                config.query_end)
    }
}