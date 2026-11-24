#!/bin/bash
# FinSight - Launch Script

set -e

echo "=========================================================="
echo " Starting FinSight: Financial Performance & Risk Platform"
echo "=========================================================="

# Activate virtual environment if present
if [ -d "venv" ]; then
    echo "Activating virtual environment (venv)..."
    source venv/bin/activate
fi

# Ensure synthetic datasets exist
if [ ! -f "data/synthetic_financial_performance.csv" ] || [ ! -f "data/synthetic_portfolio_prices.csv" ]; then
    echo "Generating synthetic datasets..."
    python3 src/data_generator.py
fi

# Run analytical verification tests
echo "Running analytical calculation tests..."
pytest tests/test_analytics.py -q

echo "Launching Streamlit application on localhost:8501..."
streamlit run app.py
