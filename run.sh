export CUDA_VISIBLE_DEVICES=0
uv run src/eval_flow.py --run-path ./checkpoints/bc --config.num_evals 50 --output-dir outputs