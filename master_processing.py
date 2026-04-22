import functions_framework
import pandas as pd
import numpy as np
import os
import io
import time
import json
import traceback
import sys
from datetime import date, datetime
from google.cloud import storage
from google.cloud.sql.connector import Connector
import sqlalchemy
from sqlalchemy import create_engine, text
import pg8000

def log_with_timestamp(message, level="INFO"):
    """Log message with timestamp and level"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    print(f"[{timestamp}] [{level}] {message}")
    sys.stdout.flush()

def log_error_with_traceback(message, error):
    """Log error with full traceback"""
    log_with_timestamp(f"ERROR: {message}", "ERROR")
    log_with_timestamp(f"Exception type: {type(error).__name__}", "ERROR")
    log_with_timestamp(f"Exception message: {str(error)}", "ERROR")
    log_with_timestamp("Full traceback:", "ERROR")
    for line in traceback.format_exc().splitlines():
        log_with_timestamp(line, "ERROR")

# Configuration
GCP_PROJECT_ID = os.environ.get("GCP_PROJECT_ID")
CLOUDSQL_INSTANCE = os.environ.get("CLOUDSQL_INSTANCE")
CLOUDSQL_DATABASE = os.environ.get("CLOUDSQL_DATABASE", "analytics_db")
CLOUDSQL_USER = os.environ.get("CLOUDSQL_USER", "postgres")
CLOUDSQL_PASSWORD = os.environ.get("CLOUDSQL_PASSWORD")

# Analytics constants
TIME_INTERVAL = "15min"
ENGAGEMENT_THRESHOLD_SECONDS = 60

# Global connector - reuse across invocations
connector = None

def get_connector():
    """Get or create global connector"""
    global connector
    if connector is None:
        log_with_timestamp("Creating new database connector...")
        connector = Connector()
        log_with_timestamp("Database connector created")
    return connector

def get_db_connection():
    """Create database connection with optimized settings"""
    log_with_timestamp("Creating database connection...")
    
    def getconn():
        conn_connector = get_connector()
        return conn_connector.connect(
            CLOUDSQL_INSTANCE,
            "pg8000",
            user=CLOUDSQL_USER,
            password=CLOUDSQL_PASSWORD,
            db=CLOUDSQL_DATABASE,
            timeout=60,
        )

    engine = create_engine(
        "postgresql+pg8000://",
        creator=getconn,
        pool_size=1,
        max_overflow=0,
        pool_pre_ping=True,
        pool_recycle=300,
        echo=False,
        connect_args={"timeout": 60}
    )
    
    # Test connection with retry logic
    max_retries = 3
    for attempt in range(max_retries):
        try:
            with engine.connect() as conn:
                result = conn.execute(text("SELECT 1"))
                result.fetchone()
            log_with_timestamp("Database connection established")
            return engine
        except Exception as e:
            log_with_timestamp(f"Connection attempt {attempt + 1}/{max_retries} failed: {str(e)}")
            if attempt == max_retries - 1:
                raise
            time.sleep(2 ** attempt)

def create_atomic_processing_lock(engine, client_id, store_id, file_name, event_id):
    """Create atomic processing lock to prevent race conditions"""
    log_with_timestamp(f"Attempting atomic lock for event: {event_id}")
    
    query = """
    INSERT INTO processing_logs (
        client_id, store_id, date, file_name, processed_at,
        raw_records, timeseries_records, zone_summaries, transitions,
        unique_zones, processing_status, error_message, processing_time_seconds, event_id
    ) VALUES (:client_id, :store_id, :date, :file_name, :processed_at,
             0, 0, 0, 0, 0, 'processing', :error_message, 0.0, :event_id)
    ON CONFLICT (client_id, store_id, file_name) 
    DO UPDATE SET 
        event_id = EXCLUDED.event_id,
        processed_at = EXCLUDED.processed_at,
        error_message = EXCLUDED.error_message
    WHERE processing_logs.processing_status IN ('failed', 'processing') 
      AND processing_logs.processed_at < NOW() - INTERVAL '10 minutes'
    RETURNING id, processing_status
    """
    
    current_time = datetime.now()
    values = {
        'client_id': client_id,
        'store_id': store_id,
        'date': current_time.date(),
        'file_name': file_name,
        'processed_at': current_time,
        'error_message': f"Event:{event_id} | Processing started at {current_time}",
        'event_id': event_id
    }
    
    try:
        with engine.connect() as conn:
            result = conn.execute(text(query), values)
            returned_row = result.fetchone()
            conn.commit()
            
            if returned_row:
                log_with_timestamp(f"Processing lock acquired (ID: {returned_row[0]}, Status: {returned_row[1] if len(returned_row) > 1 else 'new'})")
                return True
            else:
                check_query = """
                SELECT processing_status, processed_at, event_id 
                FROM processing_logs 
                WHERE client_id = :client_id AND store_id = :store_id AND file_name = :file_name
                """
                result = conn.execute(text(check_query), {
                    'client_id': client_id, 'store_id': store_id, 'file_name': file_name
                })
                existing = result.fetchone()
                
                if existing:
                    status, processed_at, existing_event = existing
                    if status == 'success':
                        log_with_timestamp(f"File already successfully processed (Event: {existing_event}, At: {processed_at})")
                    elif status == 'processing':
                        log_with_timestamp(f"File currently being processed by another instance (Event: {existing_event})")
                    else:
                        log_with_timestamp(f"File in {status} state (Event: {existing_event})")
                
                return False
                
    except Exception as e:
        log_error_with_traceback("Failed to create atomic lock", e)
        return False

def setup_database_tables(engine):
    """Create enhanced tables including first zone analytics"""
    log_with_timestamp("Setting up enhanced analytics tables...")
    
    enhanced_sql = """
    -- Create analytics_summaries table if not exists
    CREATE TABLE IF NOT EXISTS analytics_summaries (
        id SERIAL PRIMARY KEY,
        client_id VARCHAR(50) NOT NULL,
        store_id VARCHAR(50) NOT NULL,
        date DATE NOT NULL,
        processed_at TIMESTAMP NOT NULL,
        total_zones INTEGER DEFAULT 0,
        total_transitions INTEGER DEFAULT 0,
        timeseries_records INTEGER DEFAULT 0,
        engagement_threshold_seconds FLOAT DEFAULT 60,
        time_interval VARCHAR(20) DEFAULT '15min',
        file_name VARCHAR(500),
        raw_records INTEGER DEFAULT 0,
        unique_visitors INTEGER DEFAULT 0,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );

    -- Create zone_timeseries table if not exists
    CREATE TABLE IF NOT EXISTS zone_timeseries (
        id SERIAL PRIMARY KEY,
        client_id VARCHAR(50) NOT NULL,
        store_id VARCHAR(50) NOT NULL,
        date DATE NOT NULL,
        timestamp TIMESTAMP NOT NULL,
        zone VARCHAR(100) NOT NULL,
        gender VARCHAR(20),
        zone_occupancy INTEGER DEFAULT 0,
        processed_at TIMESTAMP NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );

    -- Create zone_behavioral_summaries table if not exists
    CREATE TABLE IF NOT EXISTS zone_behavioral_summaries (
        id SERIAL PRIMARY KEY,
        client_id VARCHAR(50) NOT NULL,
        store_id VARCHAR(50) NOT NULL,
        date DATE NOT NULL,
        zone VARCHAR(100) NOT NULL,
        gender VARCHAR(20),
        avg_dwell_time_seconds FLOAT DEFAULT 0,
        engaged INTEGER DEFAULT 0,
        passing INTEGER DEFAULT 0,
        total_visits INTEGER DEFAULT 0,
        engagement_rate_percent FLOAT DEFAULT 0,
        processed_at TIMESTAMP NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );

    -- Create zone_transitions table if not exists
    CREATE TABLE IF NOT EXISTS zone_transitions (
        id SERIAL PRIMARY KEY,
        client_id VARCHAR(50) NOT NULL,
        store_id VARCHAR(50) NOT NULL,
        date DATE NOT NULL,
        source_zone VARCHAR(100) NOT NULL,
        target_zone VARCHAR(100) NOT NULL,
        gender VARCHAR(20),
        transition_count INTEGER DEFAULT 0,
        processed_at TIMESTAMP NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );

    -- First zone interaction analytics table
    CREATE TABLE IF NOT EXISTS first_zone_analytics (
        id SERIAL PRIMARY KEY,
        client_id VARCHAR(50) NOT NULL,
        store_id VARCHAR(50) NOT NULL,
        camera_id VARCHAR(50) NOT NULL,
        track_id INTEGER NOT NULL,
        zone VARCHAR(100) NOT NULL,
        gender VARCHAR(20),
        interaction_timestamp TIMESTAMP NOT NULL,
        time_to_interaction_seconds FLOAT,
        processed_at TIMESTAMP NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    
    -- Enhanced occupancy tables
    CREATE TABLE IF NOT EXISTS occupancy_events (
        id SERIAL PRIMARY KEY,
        client_id VARCHAR(50) NOT NULL,
        store_id VARCHAR(50) NOT NULL,
        camera_id VARCHAR(50) NOT NULL,
        timestamp TIMESTAMP NOT NULL,
        event_type VARCHAR(10) NOT NULL CHECK (event_type IN ('entry', 'exit')),
        track_id INTEGER,
        first_zone VARCHAR(100),
        processed_at TIMESTAMP NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    
    CREATE TABLE IF NOT EXISTS occupancy_snapshots (
        id SERIAL PRIMARY KEY,
        client_id VARCHAR(50) NOT NULL,
        store_id VARCHAR(50) NOT NULL,
        timestamp TIMESTAMP NOT NULL,
        current_occupancy INTEGER DEFAULT 0,
        entries_count INTEGER DEFAULT 0,
        exits_count INTEGER DEFAULT 0,
        processed_at TIMESTAMP NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );

    -- Create processing_logs table if not exists
    CREATE TABLE IF NOT EXISTS processing_logs (
        id SERIAL PRIMARY KEY,
        client_id VARCHAR(50) NOT NULL,
        store_id VARCHAR(50) NOT NULL,
        date DATE NOT NULL,
        file_name VARCHAR(500) NOT NULL,
        processed_at TIMESTAMP NOT NULL,
        raw_records INTEGER DEFAULT 0,
        timeseries_records INTEGER DEFAULT 0,
        zone_summaries INTEGER DEFAULT 0,
        transitions INTEGER DEFAULT 0,
        unique_zones INTEGER DEFAULT 0,
        processing_status VARCHAR(20) DEFAULT 'processing',
        error_message TEXT,
        processing_time_seconds FLOAT DEFAULT 0,
        event_id VARCHAR(100),
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(client_id, store_id, file_name)
    );
    
    -- Indexes for better performance
    CREATE INDEX IF NOT EXISTS idx_first_zone_analytics_lookup ON first_zone_analytics(client_id, store_id, interaction_timestamp);
    CREATE INDEX IF NOT EXISTS idx_first_zone_analytics_zone ON first_zone_analytics(client_id, store_id, zone);
    CREATE INDEX IF NOT EXISTS idx_occupancy_events_lookup ON occupancy_events(client_id, store_id, timestamp);
    CREATE INDEX IF NOT EXISTS idx_occupancy_snapshots_lookup ON occupancy_snapshots(client_id, store_id, timestamp);
    CREATE INDEX IF NOT EXISTS idx_occupancy_snapshots_latest ON occupancy_snapshots(client_id, store_id, timestamp DESC);
    CREATE INDEX IF NOT EXISTS idx_zone_timeseries_lookup ON zone_timeseries(client_id, store_id, date);
    CREATE INDEX IF NOT EXISTS idx_zone_behavioral_lookup ON zone_behavioral_summaries(client_id, store_id, date);
    CREATE INDEX IF NOT EXISTS idx_zone_transitions_lookup ON zone_transitions(client_id, store_id, date);
    """
    
    try:
        with engine.connect() as conn:
            conn.execute(text(enhanced_sql))
            conn.commit()
        log_with_timestamp("Enhanced analytics tables created successfully")
    except Exception as e:
        log_error_with_traceback("Enhanced table creation failed", e)
        raise

def process_occupancy_data(df, client_id, store_id):
    """Enhanced occupancy processing with first zone analytics"""
    log_with_timestamp("Starting enhanced occupancy data processing...")
    if df.empty:
        log_with_timestamp("Input DataFrame is empty, returning empty occupancy data.")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    df_reset = df.reset_index()
    df_reset['timestamp'] = pd.to_datetime(df_reset['timestamp'])
    df_reset = df_reset.sort_values(by=['track_id', 'timestamp'])

    # Process first zone interactions
    first_zone_df = pd.DataFrame()
    if 'first_zone_after_entry' in df_reset.columns:
        # Filter for records with first zone interactions
        first_zone_data = df_reset[
            (df_reset['first_zone_after_entry'].notna()) & 
            (df_reset['first_zone_after_entry'] != '') &
            (df_reset.get('has_entered', True) == True)
        ].copy()
        
        if not first_zone_data.empty:
            # Get first occurrence of each track_id's first zone interaction
            first_zone_interactions = first_zone_data.groupby('track_id').first().reset_index()
            
            first_zone_df = pd.DataFrame({
                'client_id': client_id,
                'store_id': store_id,
                'camera_id': first_zone_interactions['camera_id'],
                'track_id': first_zone_interactions['track_id'],
                'zone': first_zone_interactions['first_zone_after_entry'],
                'gender': first_zone_interactions.get('gender', 'unknown'),
                'interaction_timestamp': first_zone_interactions['timestamp'],
                'time_to_interaction_seconds': 30.0,
                'processed_at': pd.Timestamp.now(tz='UTC')
            })
            
            log_with_timestamp(f"Processed {len(first_zone_df)} first zone interactions")
    
    # Process occupancy events
    occupancy_events_df = pd.DataFrame()
    
    # Try to find real entry/exit events first
    if 'crossing_status' in df_reset.columns:
        crossing_events = df_reset[
            df_reset['crossing_status'].isin(['entered', 'exited', 'entry', 'exit'])
        ].copy()
        
        if not crossing_events.empty:
            # Use real entry/exit data
            crossing_events['event_type'] = crossing_events['crossing_status'].map({
                'entered': 'entry', 'entry': 'entry',
                'exited': 'exit', 'exit': 'exit'
            })
            
            occupancy_events_df = pd.DataFrame({
                'client_id': client_id,
                'store_id': store_id,
                'camera_id': crossing_events['camera_id'],
                'timestamp': crossing_events['timestamp'],
                'event_type': crossing_events['event_type'],
                'track_id': crossing_events['track_id'],
                'first_zone': crossing_events.get('first_zone_after_entry', ''),
                'processed_at': pd.Timestamp.now(tz='UTC')
            })
            log_with_timestamp(f"Found {len(occupancy_events_df)} real entry/exit events")
    
    # If no real entry/exit events found, create synthetic ones
    if occupancy_events_df.empty and not df_reset.empty:
        log_with_timestamp("No crossing events found, creating synthetic entry events from track appearances")
        
        # Create entry events for first appearance of each track_id
        first_appearances = df_reset.groupby('track_id').first().reset_index()
        
        occupancy_events_df = pd.DataFrame({
            'client_id': client_id,
            'store_id': store_id,
            'camera_id': first_appearances['camera_id'],
            'timestamp': first_appearances['timestamp'],
            'event_type': 'entry',  # All synthetic events are entries
            'track_id': first_appearances['track_id'],
            'first_zone': first_appearances.get('first_zone_after_entry', first_appearances.get('zone', '')),
            'processed_at': pd.Timestamp.now(tz='UTC')
        })
        
        log_with_timestamp(f"Created {len(occupancy_events_df)} synthetic entry events from unique track IDs")

    # Process occupancy snapshots from events or store occupancy field
    snapshots_df = pd.DataFrame()
    
    if not occupancy_events_df.empty:
        # Create occupancy timeline from events
        log_with_timestamp("Creating occupancy snapshots from events...")
        events_sorted = occupancy_events_df.sort_values('timestamp').copy()
        events_sorted['change'] = events_sorted['event_type'].map({'entry': 1, 'exit': -1})
        events_sorted['entries_count'] = (events_sorted['change'] == 1).cumsum()
        events_sorted['exits_count'] = (events_sorted['change'] == -1).cumsum()
        events_sorted['current_occupancy'] = events_sorted['change'].cumsum()
        
        # Ensure occupancy doesn't go negative
        events_sorted['current_occupancy'] = events_sorted['current_occupancy'].clip(lower=0)
        
        snapshots_df = events_sorted[['timestamp', 'current_occupancy', 'entries_count', 'exits_count']].copy()
        snapshots_df['client_id'] = client_id
        snapshots_df['store_id'] = store_id
        snapshots_df['processed_at'] = pd.Timestamp.now(tz='UTC')
        
    elif 'store_occupancy' in df_reset.columns:
        # Create snapshots from store_occupancy field if available
        log_with_timestamp("Creating occupancy snapshots from store_occupancy field...")
        occupancy_data = df_reset[df_reset['store_occupancy'].notna()].copy()
        
        if not occupancy_data.empty:
            # Sample occupancy data at regular intervals (every 15 minutes)
            occupancy_data['time_group'] = occupancy_data['timestamp'].dt.floor('15min')
            
            # Get the latest occupancy reading for each time group
            latest_occupancy = (occupancy_data
                               .sort_values('timestamp')
                               .groupby('time_group')
                               .last()
                               .reset_index())
            
            snapshots_df = pd.DataFrame({
                'client_id': client_id,
                'store_id': store_id,
                'timestamp': latest_occupancy['time_group'],
                'current_occupancy': latest_occupancy['store_occupancy'].fillna(0).astype(int),
                'entries_count': 0,  # Would need to calculate from events
                'exits_count': 0,    # Would need to calculate from events
                'processed_at': pd.Timestamp.now(tz='UTC')
            })

    log_with_timestamp(f"Occupancy processing complete. Snapshots: {len(snapshots_df)}, Events: {len(occupancy_events_df)}, First zones: {len(first_zone_df)}")
    return snapshots_df, occupancy_events_df, first_zone_df

def save_dataframe_to_database(engine, table_name, dataframe):
    """Save DataFrame to database with batch processing"""
    if dataframe.empty:
        log_with_timestamp(f"DataFrame for {table_name} is empty, skipping")
        return
    
    log_with_timestamp(f"Saving {len(dataframe):,} rows to {table_name}")
    
    try:
        records = dataframe.to_dict('records')
        
        # Create the INSERT query with named parameters
        if table_name == 'first_zone_analytics':
            query = """
            INSERT INTO first_zone_analytics (
                client_id, store_id, camera_id, track_id, zone, gender,
                interaction_timestamp, time_to_interaction_seconds, processed_at
            ) VALUES (
                :client_id, :store_id, :camera_id, :track_id, :zone, :gender,
                :interaction_timestamp, :time_to_interaction_seconds, :processed_at
            )
            """
        
        elif table_name == 'occupancy_snapshots':
            query = """
            INSERT INTO occupancy_snapshots (
                client_id, store_id, timestamp, current_occupancy, entries_count,
                exits_count, processed_at
            ) VALUES (
                :client_id, :store_id, :timestamp, :current_occupancy, :entries_count,
                :exits_count, :processed_at
            )
            """

        elif table_name == 'occupancy_events':
            query = """
            INSERT INTO occupancy_events (
                client_id, store_id, camera_id, timestamp, event_type, 
                track_id, first_zone, processed_at
            ) VALUES (
                :client_id, :store_id, :camera_id, :timestamp, :event_type,
                :track_id, :first_zone, :processed_at
            )
            """
        
        elif table_name == 'analytics_summaries':
            query = """
            INSERT INTO analytics_summaries (
                client_id, store_id, date, processed_at, total_zones, total_transitions,
                timeseries_records, engagement_threshold_seconds, time_interval, 
                file_name, raw_records, unique_visitors
            ) VALUES (
                :client_id, :store_id, :date, :processed_at, :total_zones, :total_transitions,
                :timeseries_records, :engagement_threshold_seconds, :time_interval,
                :file_name, :raw_records, :unique_visitors
            )
            """
        
        elif table_name == 'zone_timeseries':
            query = """
            INSERT INTO zone_timeseries (
                client_id, store_id, date, timestamp, zone, gender, 
                zone_occupancy, processed_at
            ) VALUES (
                :client_id, :store_id, :date, :timestamp, :zone, :gender,
                :zone_occupancy, :processed_at
            )
            """
            
        elif table_name == 'zone_behavioral_summaries':
            query = """
            INSERT INTO zone_behavioral_summaries (
                client_id, store_id, date, zone, gender, avg_dwell_time_seconds,
                engaged, passing, total_visits, engagement_rate_percent, processed_at
            ) VALUES (
                :client_id, :store_id, :date, :zone, :gender, :avg_dwell_time_seconds,
                :engaged, :passing, :total_visits, :engagement_rate_percent, :processed_at
            )
            """
            
        elif table_name == 'zone_transitions':
            query = """
            INSERT INTO zone_transitions (
                client_id, store_id, date, source_zone, target_zone, gender,
                transition_count, processed_at
            ) VALUES (
                :client_id, :store_id, :date, :source_zone, :target_zone, :gender,
                :transition_count, :processed_at
            )
            """
        else:
            raise ValueError(f"No insert query defined for table: {table_name}")
        
        batch_size = 1000 if table_name == 'zone_timeseries' else 500
        total_inserted = 0
        
        with engine.connect() as conn:
            trans = conn.begin()
            try:
                for i in range(0, len(records), batch_size):
                    batch = records[i:i + batch_size]
                    conn.execute(text(query), batch)
                    total_inserted += len(batch)
                    
                    if total_inserted % 5000 == 0:
                        log_with_timestamp(f"Inserted batch: {total_inserted}/{len(records)} rows")
                
                trans.commit()
                log_with_timestamp(f"Successfully saved {total_inserted:,} rows to {table_name}")
                
            except Exception as e:
                trans.rollback()
                raise
        
    except Exception as e:
        log_error_with_traceback(f"Failed to save to {table_name}", e)
        raise

# Main processing function
@functions_framework.cloud_event
def master_data_processor(cloud_event):
    """Main data processing function with occupancy and first zone analytics"""
    start_time = datetime.now()
    
    log_with_timestamp("=" * 80)
    log_with_timestamp("DATA PROCESSING STARTED")
    log_with_timestamp(f"Start time: {start_time}")
    log_with_timestamp("=" * 80)
    
    engine = None
    
    try:
        # Extract event information
        bucket_name = cloud_event.data.get("bucket", "unknown")
        file_name = cloud_event.data.get("name", "unknown")
        event_id = cloud_event.data.get("eventId", f"manual-{int(start_time.timestamp())}")
        
        log_with_timestamp(f"Event: {event_id}")
        log_with_timestamp(f"Bucket: {bucket_name}")
        log_with_timestamp(f"File: {file_name}")
        
        # Validate file
        if not file_name.endswith('.parquet') or not file_name.startswith('staging_zone/'):
            log_with_timestamp("SKIPPED: File doesn't match criteria")
            return {"status": "skipped", "reason": "File doesn't match criteria"}
        
        # Parse metadata from file path
        path_parts = file_name.split('/')
        metadata_parts = {}
        for part in path_parts:
            if '=' in part:
                key, value = part.split('=', 1)
                metadata_parts[key] = value
        
        client_id = metadata_parts.get('client_id')
        store_id = metadata_parts.get('store_id')
        
        if not client_id or not store_id:
            log_with_timestamp("ERROR: Could not parse client_id or store_id from path", "ERROR")
            return {"status": "error", "reason": "Could not parse client_id or store_id"}
        
        log_with_timestamp(f"Client: {client_id}, Store: {store_id}")
        
        # Setup database
        engine = get_db_connection()
        setup_database_tables(engine)
        
        # Create atomic processing lock
        if not create_atomic_processing_lock(engine, client_id, store_id, file_name, event_id):
            log_with_timestamp("PROCESSING STOPPED - File already processed or being processed")
            return {"status": "skipped", "reason": "File already processed or locked"}
        
        log_with_timestamp("Processing lock acquired - proceeding with analytics")
        
        # Read parquet file from Google Cloud Storage
        log_with_timestamp("Reading parquet file...")
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(file_name)

        if not blob.exists():
            log_with_timestamp(f"ERROR: File {file_name} does not exist in bucket {bucket_name}", "ERROR")
            return {"status": "error", "reason": "File does not exist"}
        
        try:
            parquet_content = blob.download_as_bytes()
            raw_df = pd.read_parquet(io.BytesIO(parquet_content))
        except Exception as download_error:
            log_with_timestamp(f"ERROR: Failed to download or parse parquet file: {str(download_error)}", "ERROR")
            raise
        
        log_with_timestamp(f"Loaded {len(raw_df):,} records from parquet file")
        
        if raw_df.empty:
            log_with_timestamp("WARNING: Empty dataset - skipping processing")
            return {"status": "skipped", "reason": "Empty dataset"}
        
        # Process timestamps
        log_with_timestamp("Processing timestamps...")
        raw_df['timestamp'] = pd.to_datetime(raw_df['timestamp'])
        processing_date = raw_df["timestamp"].iloc[0].date()
        raw_records_count = len(raw_df)
        raw_df.set_index('timestamp', inplace=True)
        
        # Calculate zone timeseries
        log_with_timestamp("Calculating zone timeseries...")
        df_for_timeseries = raw_df.reset_index()
        df_for_timeseries['time_group'] = df_for_timeseries['timestamp'].dt.floor(TIME_INTERVAL)

        latest_positions = (df_for_timeseries
                           .sort_values('timestamp')
                           .groupby(['time_group', 'camera_id', 'track_id'])
                           .last()
                           .reset_index())

        zone_timeseries_df = (latest_positions
                             .groupby(['time_group', 'zone', 'gender'])
                             ['track_id']
                             .nunique()
                             .reset_index()
                             .rename(columns={'track_id': 'zone_occupancy', 'time_group': 'timestamp'}))

        zone_timeseries_df['store_id'] = store_id
        zone_timeseries_df['client_id'] = client_id
        zone_timeseries_df['date'] = processing_date
        zone_timeseries_df['processed_at'] = datetime.now()

        log_with_timestamp(f"Generated {len(zone_timeseries_df):,} timeseries records")

        # Calculate behavioral summaries
        log_with_timestamp("Calculating behavioral summaries...")
        df_for_dwell_time = raw_df.reset_index()
        df_for_dwell_time = df_for_dwell_time.sort_values(['camera_id', 'track_id', 'timestamp'])

        df_for_dwell_time['time_diff'] = (df_for_dwell_time
                                         .groupby(['camera_id', 'track_id'])['timestamp']
                                         .diff().dt.total_seconds().fillna(0))

        df_for_dwell_time['is_new_zone'] = (
            (df_for_dwell_time['zone'] != df_for_dwell_time['zone'].shift(1)) | 
            (df_for_dwell_time['track_id'] != df_for_dwell_time['track_id'].shift(1)) |
            (df_for_dwell_time['camera_id'] != df_for_dwell_time['camera_id'].shift(1))
        )
        df_for_dwell_time['zone_session_id'] = df_for_dwell_time['is_new_zone'].cumsum()

        zone_sessions = (df_for_dwell_time
                        .groupby(['zone', 'gender', 'camera_id', 'track_id', 'zone_session_id'])
                        .agg({
                            'time_diff': 'sum',
                            'timestamp': 'count'
                        })
                        .reset_index()
                        .rename(columns={'time_diff': 'session_duration', 'timestamp': 'detection_count'}))

        zone_sessions = zone_sessions[zone_sessions['detection_count'] >= 2]

        person_zone_totals = (zone_sessions
                             .groupby(['zone', 'gender', 'camera_id', 'track_id'])
                             ['session_duration']
                             .sum()
                             .reset_index()
                             .rename(columns={'session_duration': 'total_dwell_time'}))

        person_zone_totals['visit_type'] = np.where(
            person_zone_totals['total_dwell_time'] >= ENGAGEMENT_THRESHOLD_SECONDS, 
            'Engaged', 
            'Passing'
        )

        avg_dwell_time = (person_zone_totals.groupby(['zone', 'gender'])['total_dwell_time']
                         .mean().reset_index()
                         .rename(columns={'total_dwell_time': 'avg_dwell_time_seconds'}))

        engagement_summary = (person_zone_totals.groupby(['zone', 'gender'])['visit_type']
                            .value_counts().unstack(fill_value=0))

        if 'Engaged' not in engagement_summary.columns:
            engagement_summary['Engaged'] = 0
        if 'Passing' not in engagement_summary.columns:
            engagement_summary['Passing'] = 0

        engagement_summary['total_visits'] = (engagement_summary['Engaged'] + 
                                            engagement_summary['Passing'])
        engagement_summary['engagement_rate_percent'] = (
            (engagement_summary['Engaged'] / engagement_summary['total_visits'] * 100)
            .round(2)
        )

        zone_summary_df = pd.merge(avg_dwell_time, engagement_summary.reset_index(), 
                                  on=['zone', 'gender'])
        zone_summary_df['store_id'] = store_id
        zone_summary_df['date'] = processing_date
        zone_summary_df['client_id'] = client_id
        zone_summary_df['processed_at'] = datetime.now()
        zone_summary_df = zone_summary_df.rename(columns={'Engaged': 'engaged', 'Passing': 'passing'})

        log_with_timestamp(f"Generated {len(zone_summary_df):,} zone behavioral summaries")

        # Calculate zone transitions
        log_with_timestamp("Calculating zone transitions...")
        df_for_transitions = raw_df.reset_index()
        df_for_transitions = df_for_transitions.sort_values(['camera_id', 'track_id', 'timestamp'])

        df_for_transitions['target_zone'] = (df_for_transitions
                                           .groupby(['camera_id', 'track_id'])['zone']
                                           .shift(-1))

        transitions = df_for_transitions[
            (df_for_transitions['zone'] != df_for_transitions['target_zone']) & 
            (df_for_transitions['target_zone'].notna())
        ].copy()

        transitions_df = (transitions.groupby(['zone', 'gender', 'target_zone'])
                         ['track_id']
                         .nunique()
                         .reset_index(name='transition_count')
                         .rename(columns={'zone': 'source_zone'}))

        transitions_df['store_id'] = store_id
        transitions_df['date'] = processing_date
        transitions_df['client_id'] = client_id
        transitions_df['processed_at'] = datetime.now()

        log_with_timestamp(f"Generated {len(transitions_df):,} zone transitions")

        # Calculate summary metrics
        unique_visitors_count = raw_df.reset_index()['track_id'].nunique()
        unique_zones = len(zone_summary_df['zone'].unique()) if not zone_summary_df.empty else 0

        summary_df = pd.DataFrame([{
            'client_id': client_id,
            'store_id': store_id,
            'date': processing_date,
            'processed_at': datetime.now(),
            'total_zones': unique_zones,
            'total_transitions': len(transitions_df),
            'timeseries_records': len(zone_timeseries_df),
            'engagement_threshold_seconds': ENGAGEMENT_THRESHOLD_SECONDS,
            'time_interval': TIME_INTERVAL,
            'file_name': file_name,
            'raw_records': raw_records_count,
            'unique_visitors': unique_visitors_count
        }])

        log_with_timestamp(f"Summary: {raw_records_count:,} raw records -> {unique_visitors_count} unique visitors")
        
        # Process occupancy and first zone analytics
        log_with_timestamp("Processing occupancy and first zone analytics...")
        occupancy_snapshots_df, occupancy_events_df, first_zone_df = process_occupancy_data(
            raw_df, client_id, store_id
        )

        # Save all data to database
        log_with_timestamp("Saving to database...")
        
        save_dataframe_to_database(engine, 'analytics_summaries', summary_df)
        save_dataframe_to_database(engine, 'zone_timeseries', zone_timeseries_df)
        save_dataframe_to_database(engine, 'zone_behavioral_summaries', zone_summary_df)
        save_dataframe_to_database(engine, 'zone_transitions', transitions_df)

        # Save occupancy data
        if not occupancy_snapshots_df.empty:
            save_dataframe_to_database(engine, 'occupancy_snapshots', occupancy_snapshots_df)
        
        if not occupancy_events_df.empty:
            save_dataframe_to_database(engine, 'occupancy_events', occupancy_events_df)
        
        if not first_zone_df.empty:
            save_dataframe_to_database(engine, 'first_zone_analytics', first_zone_df)
        
        # Update processing status
        log_with_timestamp("Updating processing status...")
        query = """
        UPDATE processing_logs 
        SET processing_status = 'success',
            raw_records = :raw_records,
            timeseries_records = :timeseries_records,
            zone_summaries = :zone_summaries,
            transitions = :transitions,
            unique_zones = :unique_zones,
            processing_time_seconds = :processing_time_seconds,
            processed_at = :processed_at
        WHERE client_id = :client_id 
          AND store_id = :store_id 
          AND file_name = :file_name 
          AND event_id = :event_id
        """
        
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        
        with engine.connect() as conn:
            conn.execute(text(query), {
                'raw_records': raw_records_count,
                'timeseries_records': len(zone_timeseries_df),
                'zone_summaries': len(zone_summary_df),
                'transitions': len(transitions_df),
                'unique_zones': unique_zones,
                'processing_time_seconds': processing_time,
                'processed_at': end_time,
                'client_id': client_id,
                'store_id': store_id,
                'file_name': file_name,
                'event_id': event_id
            })
            conn.commit()
        
        log_with_timestamp("=" * 80)
        log_with_timestamp("DATA PROCESSING COMPLETED SUCCESSFULLY!")
        log_with_timestamp(f"Processing time: {processing_time:.2f}s")
        log_with_timestamp(f"Raw records: {raw_records_count:,}")
        log_with_timestamp(f"First zone interactions: {len(first_zone_df)}")
        log_with_timestamp(f"Occupancy events: {len(occupancy_events_df)}")
        log_with_timestamp(f"Event ID: {event_id}")
        log_with_timestamp("=" * 80)
        
        return {
            "status": "success",
            "processing_time": processing_time,
            "raw_records": raw_records_count,
            "first_zone_interactions": len(first_zone_df),
            "occupancy_events": len(occupancy_events_df),
            "event_id": event_id,
            "client_id": client_id,
            "store_id": store_id
        }
        
    except Exception as e:
        error_msg = f"Processing failed: {str(e)}"
        log_with_timestamp("DATA PROCESSING FAILED!")
        log_error_with_traceback("Main processing error", e)
        
        # Update processing status to failed
        if engine and 'client_id' in locals() and 'store_id' in locals() and 'file_name' in locals() and 'event_id' in locals():
            try:
                query = """
                UPDATE processing_logs 
                SET processing_status = 'failed',
                    error_message = :error_message,
                    processing_time_seconds = :processing_time_seconds
                WHERE client_id = :client_id 
                  AND store_id = :store_id 
                  AND file_name = :file_name 
                  AND event_id = :event_id
                """
                
                processing_time = (datetime.now() - start_time).total_seconds()
                
                with engine.connect() as conn:
                    conn.execute(text(query), {
                        'error_message': error_msg,
                        'processing_time_seconds': processing_time,
                        'client_id': client_id,
                        'store_id': store_id,
                        'file_name': file_name,
                        'event_id': event_id
                    })
                    conn.commit()
                    
                log_with_timestamp("Processing status updated to failed")
            except Exception as update_error:
                log_with_timestamp(f"Failed to update processing status: {str(update_error)}")
        
        return {"status": "error", "error": error_msg}
    
    finally:
        # Clean up database connection
        if engine:
            try:
                engine.dispose()
                log_with_timestamp("Database connection disposed")
            except Exception as e:
                log_with_timestamp(f"Warning: Failed to dispose engine: {str(e)}")