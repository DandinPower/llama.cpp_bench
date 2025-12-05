#!/usr/bin/env bash
set -euo pipefail

###############################################################################
# CONFIG SECTION - EDIT THESE VALUES ONLY
###############################################################################

# Path to llama-bench binary
LLAMA_BENCH="./llama-bench"

# Models to test (add/remove models here)
MODELS=(
  "ggufs/Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf"
  "ggufs/Qwen3-32B-Q4_K_M.gguf"
  "ggufs/Qwen3-8B-Q4_K_M.gguf"
  "ggufs/Qwen3-1.7B-Q4_K_M.gguf"
  "ggufs/Ministral-3-14B-Instruct-2512-Q4_K_M.gguf"
  "ggufs/Ministral-3-8B-Instruct-2512-Q4_K_M.gguf"
  "ggufs/Ministral-3-3B-Instruct-2512-Q4_K_M.gguf"
)

# Global benchmarking settings
REPS=5          # -r, repetitions per test
THREADS=20      # -t, number of threads
OUTPUT_FORMAT="md"    # -o, one of: md, csv, json, jsonl, sql

# GPU settings - force GPU usage
GPU_LAYERS=99   # -ngl, number of layers offloaded to GPU
MAIN_GPU=0      # -mg, GPU index

# Prompt processing (prefill) tests
RUN_PP_TESTS=1          # set to 0 to disable pp tests
PP_CONTEXT_LENGTHS=(      # -p values (pp only)
  16
  32
  64
  128
  256
  512
  1024
  2048
  4096
  8192
  16384
)
PP_N_GEN=0              # -n, 0 for pure pp
PP_BATCH_SIZE=16384       # -b for pp
PP_UBATCH_SIZE=16384      # -ub for pp

# Decoding tests at some context depth
RUN_DEC_TESTS=1             # set to 0 to disable decode tests
# --- NEW: Arrays for the desired combinations ---
DEC_DEPTHS=(512 2048)      # -d, context depth values
DEC_N_GENS=(64 256 512)    # -n, tokens to generate values
# ------------------------------------------------
DEC_BATCH_SIZE=1         # -b, batch size during decoding
DEC_UBATCH_SIZES=(         # -ub values for decoding
  1
)

# Extra llama-bench options if you want to tweak more parameters
# For example: EXTRA_ARGS="--flash-attn 1"
EXTRA_ARGS="--mmap 0 --flash-attn 1"

###############################################################################
# INTERNAL HELPER FUNCTIONS
###############################################################################

join_by_comma() {
  local IFS=,
  echo "$*"
}

run_pp_tests_for_model() {
  local model="$1"
  local ctx_csv
  ctx_csv=$(join_by_comma "${PP_CONTEXT_LENGTHS[@]}")

  echo "------------------------------------------------------------"
  echo "Model: $model"
  echo "PP tests: n_prompt = $ctx_csv, n_gen = $PP_N_GEN"
  echo "------------------------------------------------------------"

  "$LLAMA_BENCH" \
    -m "$model" \
    -r "$REPS" \
    -t "$THREADS" \
    -ngl "$GPU_LAYERS" \
    -mg "$MAIN_GPU" \
    -p "$ctx_csv" \
    -n "$PP_N_GEN" \
    -b "$PP_BATCH_SIZE" \
    -ub "$PP_UBATCH_SIZE" \
    -o "$OUTPUT_FORMAT" \
    $EXTRA_ARGS
}

# --- MODIFIED: function now takes depth and n_gen as arguments ---
run_dec_tests_for_model() {
  local model="$1"
  local dec_depth="$2" # New argument
  local dec_n_gen="$3" # New argument
  local ub_csv
  ub_csv=$(join_by_comma "${DEC_UBATCH_SIZES[@]}")

  echo
  echo "------------------------------------------------------------"
  echo "Model: $model"
  echo "Decode tests: tg$dec_n_gen @ d$dec_depth, ubatch = $ub_csv"
  echo "------------------------------------------------------------"

  "$LLAMA_BENCH" \
    -m "$model" \
    -r "$REPS" \
    -t "$THREADS" \
    -ngl "$GPU_LAYERS" \
    -mg "$MAIN_GPU" \
    -p 0 \
    -n "$dec_n_gen" \
    -d "$dec_depth" \
    -b "$DEC_BATCH_SIZE" \
    -ub "$ub_csv" \
    -o "$OUTPUT_FORMAT" \
    $EXTRA_ARGS
}
# ------------------------------------------------------------------

###############################################################################
# MAIN
###############################################################################

# Sanity checks
if [ ! -x "$LLAMA_BENCH" ]; then
  echo "Error: llama-bench binary not found or not executable at: $LLAMA_BENCH" >&2
  exit 1
fi

if [ "${#MODELS[@]}" -eq 0 ]; then
  echo "Error: MODELS array is empty. Add at least one model path in CONFIG section." >&2
  exit 1
fi

echo "============================================================"
echo "llama-bench multi model benchmark"
echo
echo "Repetitions : $REPS"
echo "Threads     : $THREADS"
echo "GPU layers  : $GPU_LAYERS"
echo "Main GPU    : $MAIN_GPU"
echo "Output      : $OUTPUT_FORMAT"
echo "PP tests    : ${RUN_PP_TESTS}"
echo "DEC tests   : ${RUN_DEC_TESTS}"
echo "Decode Depths : ${DEC_DEPTHS[*]}"
echo "Decode N Gen  : ${DEC_N_GENS[*]}"
echo "============================================================"
echo

for MODEL in "${MODELS[@]}"; do
  if [ ! -f "$MODEL" ]; then
    echo "Warning: model file not found, skipping: $MODEL" >&2
    continue
  fi

  if [ "$RUN_PP_TESTS" -eq 1 ]; then
    run_pp_tests_for_model "$MODEL"
    echo
  fi

  if [ "$RUN_DEC_TESTS" -eq 1 ]; then
    # --- NEW: Nested loops for combinations ---
    for DEPTH in "${DEC_DEPTHS[@]}"; do
      for N_GEN in "${DEC_N_GENS[@]}"; do
        run_dec_tests_for_model "$MODEL" "$DEPTH" "$N_GEN"
        echo
      done
    done
    # ------------------------------------------
  fi

  echo
done

echo "All requested tests finished."
