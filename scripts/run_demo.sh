#!/bin/bash

curl -X POST "http://127.0.0.1:8000/recommend" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "AE23ZBUF2YVBQPH2NN6F5XSA3QYQ",
    "top_k": 10,
    "mode": "genrec_gru"
  }'