#!/usr/bin/env bash
source venv/bin/activate
python openai_api_server.py --gpus=0,1,2,3 --base_model=01-ai/Yi-34B-Chat