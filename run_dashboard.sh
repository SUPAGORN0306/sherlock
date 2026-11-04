#!/usr/bin/env bash
cd /home/pleum/projects/fraud-detection/sherlock
/home/pleum/miniconda3/envs/fraud-env/bin/streamlit run src/dashboard/app.py --server.port 8501 --server.address 0.0.0.0
