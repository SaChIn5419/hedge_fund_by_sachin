#!/bin/bash
# Chimera Daily Telemetry Pipeline
# Purpose: Fetch news, execute Dexter feature build, and run the engine for forward telemetry.
# Schedule: Daily 09:05 AM IST

set -e

# API Configurations
export NEWSAPI_KEY="c204b44b62b048c6a4842098f84ac847"
export NEWSDATAIO_API_KEY="pub_587faa8ad2024983b2c63eacee63ec43"
export GNEWS_API_KEY="29d4f928854c0054b96ccc3aa1b31436"

# Paths
ROOT_DIR="/home/sachindb/Documents/hedgefund_chimera"
LOG_FILE="$ROOT_DIR/data/telemetry/pipeline_log.txt"
DATE_TODAY=$(date +%Y-%m-%d)

mkdir -p "$ROOT_DIR/data/telemetry"

echo "[$DATE_TODAY $(date +%T)] Starting Chimera Daily Pipeline..." >> "$LOG_FILE"

cd "$ROOT_DIR"

# 1. News Ingestion (9:00 AM IST Cutoff)
echo "Step 1/3: Fetching News..." >> "$LOG_FILE"
if python3 -m research.experiments.build_newsapi_articles --from-date "$DATE_TODAY" --to-date "$DATE_TODAY" --cutoff-time-ist 09:00 >> "$LOG_FILE" 2>&1; then
    echo "SUCCESS: News Ingestion" >> "$LOG_FILE"
else
    echo "FAILURE: News Ingestion" >> "$LOG_FILE"
    exit 1
fi

# 2. Dexter Feature Generation
echo "Step 2/3: Building Dexter Features..." >> "$LOG_FILE"
if python3 -m research.experiments.build_dexter_features >> "$LOG_FILE" 2>&1; then
    echo "SUCCESS: Dexter Feature Build" >> "$LOG_FILE"
else
    echo "FAILURE: Dexter Feature Build" >> "$LOG_FILE"
    exit 1
fi

# 3. Engine Execution (Telemetry Run)
echo "Step 3/3: Running Engine for Telemetry..." >> "$LOG_FILE"
if python3 -m engine.signal >> "$LOG_FILE" 2>&1; then
    echo "SUCCESS: Engine Execution" >> "$LOG_FILE"
else
    echo "FAILURE: Engine Execution" >> "$LOG_FILE"
    exit 1
fi

# 4. Regime Heartbeat Update
echo "Step 4/4: Updating Regime Heartbeat..." >> "$LOG_FILE"
if python3 scripts/telemetry/regime_heartbeat.py >> "$LOG_FILE" 2>&1; then
    echo "SUCCESS: Heartbeat Updated" >> "$LOG_FILE"
else
    echo "FAILURE: Heartbeat Update" >> "$LOG_FILE"
    # Non-fatal error, we don't exit 1 here as the engine run succeeded
fi

echo "[$DATE_TODAY $(date +%T)] Pipeline Completed Successfully." >> "$LOG_FILE"
