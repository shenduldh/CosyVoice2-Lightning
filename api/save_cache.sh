IP=192.168.0.90
PORT=12244

python save_cache.py \
    --ip $IP --port $PORT \
    --cache_dir ../assets \
    --prompt_ids teemo,twitch