#!/bin/bash
# Kill any existing streamlit processes before starting fresh
pkill -f "streamlit run app.py" 2>/dev/null
sleep 1
cd "$(dirname "$0")"
python3 -m streamlit run app.py --server.headless true --browser.gatherUsageStats false
