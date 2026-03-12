#!/bin/bash
set -e

export HF_HOME=/home/dongjig/.cache/hf_home
export HF_DATASETS_CACHE=/home/dongjig/.cache/hf_datasets_cache
export VLLM_PLUGINS=nemotron_nano_asr
export PYTHONPATH=/home/dongjig/Skills:$PYTHONPATH

echo "=== Starting vLLM ASR server ==="
conda run -n nemo python -m nemo_skills.inference.server.serve_unified \
  --backend vllm_asr \
  --model /home/dongjig/nemotron-nano-asr-ckpt \
  --tokenizer nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 \
  --host 0.0.0.0 --port 8000 &
SERVER_PID=$!

echo "Waiting for server to start (PID=$SERVER_PID)..."
for i in $(seq 1 120); do
  if curl -s http://localhost:8000/ > /dev/null 2>&1; then
    echo "Server is ready!"
    break
  fi
  sleep 5
done

if ! curl -s http://localhost:8000/ > /dev/null 2>&1; then
  echo "Server failed to start"
  kill $SERVER_PID 2>/dev/null
  exit 1
fi

echo ""
echo "=== Running evaluations ==="

DATASETS=(
  "ami test"
  "earnings22 test"
  "gigaspeech test"
  "librispeech test.clean"
  "librispeech test.other"
  "spgispeech test"
  "tedlium test"
  "voxpopuli test"
)

for entry in "${DATASETS[@]}"; do
  DATASET=$(echo $entry | cut -d' ' -f1)
  SPLIT=$(echo $entry | cut -d' ' -f2)
  echo ""
  echo "=== Evaluating $DATASET $SPLIT ==="
  conda run -n nemo python test_vllm_asr_librispeech.py \
    --dataset "$DATASET" --split "$SPLIT" --concurrency 32 || true
done

echo ""
echo "=== All evaluations complete ==="
echo "Results saved in /home/dongjig/results/vllm_asr_eval/"
ls -la /home/dongjig/results/vllm_asr_eval/

echo "Stopping server..."
kill $SERVER_PID 2>/dev/null
wait $SERVER_PID 2>/dev/null
echo "Done."
