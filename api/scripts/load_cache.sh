IP=localhost
PORT=12244

python scripts/load_cache.py \
    --ip $IP --port $PORT \
    --cache_path ../assets/speaker_cache_example.pt