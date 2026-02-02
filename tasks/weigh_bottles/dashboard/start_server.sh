#!/bin/bash
# Start the AURA Dashboard Server
cd /home/mani/Repos/aura
source .venv/bin/activate
exec python tasks/weigh_bottles/dashboard/run_server.py
