use dotenvy::dotenv;
use std::env;
use std::error::Error;

pub struct AppConfig {
    pub db_url: String,
    pub query_instrument: String,
    pub query_timeframe: String,
    pub query_start: String,
    pub query_end: String,
}

impl AppConfig {
    pub fn load() -> Result<Self, Box<dyn Error>> {
        dotenv().ok();

        let db_host = env::var("CLICKHOUSE_HOST").expect("HOST not set");
        let db_port = env::var("CLICKHOUSE_PORT").unwrap_or_else(|_| "9000".to_string());
        let db_user = env::var("CLICKHOUSE_USER").unwrap_or_else(|_| "default".to_string());
        let db_pass = env::var("CLICKHOUSE_PASSWORD").unwrap_or_else(|_| "".to_string());
        let db_name = env::var("CLICKHOUSE_DB").expect("DB name not set");
        let db_url = format!(
            "tcp://{}:{}@{}:{}/{}",
            db_user, db_pass, db_host, db_port, db_name
        );

        let q_instrument = Self::validate_input(
            &env::var("QUERY_INSTRUMENT").unwrap_or_else(|_| "EURUSD".to_string()),
            "Instrument",
        )?;
        let q_timeframe = Self::validate_input(
            &env::var("QUERY_TIMEFRAME").unwrap_or_else(|_| "m1".to_string()),
            "Timeframe",
        )?;
        let q_start = Self::validate_input(
            &env::var("QUERY_START_DATE").expect("START DATE not set"),
            "Start Date",
        )?;
        let q_end = Self::validate_input(
            &env::var("QUERY_END_DATE").expect("END DATE not set"),
            "End Date",
        )?;

        Ok(Self {
            db_url,
            query_instrument: q_instrument,
            query_timeframe: q_timeframe,
            query_start: q_start,
            query_end: q_end,
        })
    }

    fn validate_input(input: &str, field_name: &str) -> Result<String, Box<dyn Error>> {
        if input
            .chars()
            .all(|c| c.is_alphanumeric() || c == '-' || c == '_' || c == ' ' || c == ':')
        {
            Ok(input.to_string())
        } else {
            Err(format!("Invalid characters in {}: potentially unsafe.", field_name).into())
        }
    }
}
