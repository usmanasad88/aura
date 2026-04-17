#!/bin/bash
# AURA Experiment Runner — systematically run experiments with different configurations.
#
# Logs are organized by experiment ID (derived from CLI args). Re-running with the same
# settings creates a new repetition under the same experiment directory, unless --fresh
# is passed to start clean.
#
# Usage:
#   ./scripts/run_experiments.sh                # Run all experiments
#   ./scripts/run_experiments.sh --fresh        # Clear previous results and re-run
#   ./scripts/run_experiments.sh --eval-only    # Only evaluate existing logs
#   ./scripts/run_experiments.sh --tier 1       # Run specific tier only
#
# Directory structure:
#   logs/experiments/<experiment_id>/
#     manifest.json            # experiment config
#     rep_001/                 # first run
#       intent_monitor/        # moved from session logs
#       decision_engine/
#       run_results.json       # evaluation output
#     rep_002/                 # second run (auto-incremented)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AURA_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$AURA_ROOT"

PYTHON=".venv/bin/python"
EXPERIMENTS_DIR="logs/experiments"

# ── Configuration ─────────────────────────────────────────────────────────
TASK="hand_layup"
VIDEO="demo_data/layup_demo/layup_gesture_demo_stationary_with_overlay.mp4"
BASE_FLAGS="--no-realtime --no-dashboard --dry-run"
MAX_CYCLES=300
FRAME_SKIP=90

# Models for Tier 1
#MODELS=("gemini-2.5-flash" "gemini-2.5-pro" "gemini-3.1-pro-preview")
#MODELS=("gemini-3.1-pro-preview", "gemini-3.1-flash-lite-preview", "gemini-3-flash-preview")
MODELS=("gemini-3.1-flash-lite-preview")
REPS=1

# ── Argument parsing ──────────────────────────────────────────────────────
FRESH=false
EVAL_ONLY=false
TIER="all"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --fresh)    FRESH=true; shift ;;
        --eval-only) EVAL_ONLY=true; shift ;;
        --tier)     TIER="$2"; shift 2 ;;
        --reps)     REPS="$2"; shift 2 ;;
        --task)     TASK="$2"; shift 2 ;;
        --video)    VIDEO="$2"; shift 2 ;;
        *)          echo "Unknown option: $1"; exit 1 ;;
    esac
done

mkdir -p "$EXPERIMENTS_DIR"

# ── Helper functions ──────────────────────────────────────────────────────

# Generate experiment ID from parameters
exp_id() {
    local task="$1" model="$2" fs="$3" gt_flag="$4"
    local model_slug="${model//\//-}"
    model_slug="${model_slug// /-}"
    local id="${task}__${model_slug}__fs${fs}"
    if [[ "$gt_flag" == "true" ]]; then
        id="${id}__gt"
    fi
    echo "$id"
}

# Find next repetition number
next_rep() {
    local exp_dir="$1"
    local n=1
    while [[ -d "$exp_dir/rep_$(printf '%03d' $n)" ]]; do
        n=$((n + 1))
    done
    printf '%03d' "$n"
}

# Run a single experiment
run_single() {
    local task="$1" model="$2" frame_skip="$3" gt_robot="$4"
    local extra_flags="${5:-}"

    local eid
    eid=$(exp_id "$task" "$model" "$frame_skip" "$gt_robot")
    local exp_dir="$EXPERIMENTS_DIR/$eid"

    if [[ "$FRESH" == "true" ]] && [[ -d "$exp_dir" ]]; then
        echo "  [FRESH] Removing previous results: $exp_dir"
        rm -rf "$exp_dir"
    fi

    mkdir -p "$exp_dir"

    # Save/update manifest
    cat > "$exp_dir/manifest.json" <<MANIFEST
{
  "experiment_id": "$eid",
  "task": "$task",
  "model": "$model",
  "frame_skip": $frame_skip,
  "ground_truth_robot": $gt_robot,
  "video": "$VIDEO",
  "base_flags": "$BASE_FLAGS",
  "max_cycles": $MAX_CYCLES,
  "extra_flags": "$extra_flags"
}
MANIFEST

    if [[ "$EVAL_ONLY" == "true" ]]; then
        echo "  [EVAL-ONLY] Skipping run, evaluating existing reps..."
        evaluate_all_reps "$exp_dir" "$task"
        return
    fi

    local rep
    rep=$(next_rep "$exp_dir")
    local rep_dir="$exp_dir/rep_$rep"
    mkdir -p "$rep_dir"

    echo "  Run: $eid  rep=$rep"

    # Clear session logs before run so we can find the new ones
    local ts_before
    ts_before=$(date +%Y%m%d_%H%M%S)

    # Build flags
    local gt_flag=""
    if [[ "$gt_robot" == "true" ]]; then
        gt_flag="--use-ground-truth-robot-status"
    fi

    # Run AURA
    set +e
    $PYTHON scripts/run_aura.py \
        --task "$task" \
        --video "$VIDEO" \
        --model "$model" \
        --no-realtime --no-dashboard --dry-run \
        $gt_flag \
        --frame-skip "$frame_skip" \
        --max-cycles "$MAX_CYCLES" \
        $extra_flags \
        2>&1 | tee "$rep_dir/run.log"
    local exit_code=$?
    set -e

    echo "$exit_code" > "$rep_dir/exit_code"

    if [[ $exit_code -ne 0 ]]; then
        echo "  [WARN] Run exited with code $exit_code"
    fi

    # Move latest session logs into rep directory
    move_latest_sessions "$rep_dir" "$ts_before"

    # Evaluate this rep
    evaluate_rep "$rep_dir" "$task"
}

# Move newly created session logs into the repetition directory
move_latest_sessions() {
    local rep_dir="$1" ts_before="$2"

    # Find intent monitor sessions created after ts_before
    for sdir in logs/intent_monitor/session_*; do
        if [[ -d "$sdir" ]]; then
            local sname
            sname=$(basename "$sdir")
            local sts="${sname#session_}"
            if [[ "$sts" > "$ts_before" ]] || [[ "$sts" == "$ts_before" ]]; then
                mkdir -p "$rep_dir/intent_monitor"
                mv "$sdir" "$rep_dir/intent_monitor/$sname"
                echo "  Moved $sname → rep/intent_monitor/"
            fi
        fi
    done

    # Find decision engine sessions
    for sdir in logs/decision_engine/session_*; do
        if [[ -d "$sdir" ]]; then
            local sname
            sname=$(basename "$sdir")
            local sts="${sname#session_}"
            if [[ "$sts" > "$ts_before" ]] || [[ "$sts" == "$ts_before" ]]; then
                mkdir -p "$rep_dir/decision_engine"
                mv "$sdir" "$rep_dir/decision_engine/$sname"
                echo "  Moved $sname → rep/decision_engine/"
            fi
        fi
    done
}

# Evaluate a single repetition
evaluate_rep() {
    local rep_dir="$1" task="$2"

    # Find the session directories inside the rep
    local intent_session="" decision_session=""

    if [[ -d "$rep_dir/intent_monitor" ]]; then
        intent_session=$(find "$rep_dir/intent_monitor" -maxdepth 1 -name "session_*" -type d | sort | tail -1)
    fi
    if [[ -d "$rep_dir/decision_engine" ]]; then
        decision_session=$(find "$rep_dir/decision_engine" -maxdepth 1 -name "session_*" -type d | sort | tail -1)
    fi

    local eval_args="--task $task"
    if [[ -n "$intent_session" ]]; then
        eval_args="$eval_args --intent-session $intent_session"
    fi
    if [[ -n "$decision_session" ]]; then
        eval_args="$eval_args --decision-session $decision_session"
    fi

    if [[ -n "$intent_session" ]] || [[ -n "$decision_session" ]]; then
        echo "  Evaluating rep..."
        $PYTHON scripts/eval/evaluate_run.py $eval_args \
            --output "$rep_dir/run_results.json" 2>&1 | sed 's/^/    /'
    else
        echo "  [WARN] No session logs found in $rep_dir"
    fi
}

# Evaluate all reps in an experiment directory
evaluate_all_reps() {
    local exp_dir="$1" task="$2"
    for rep_dir in "$exp_dir"/rep_*; do
        if [[ -d "$rep_dir" ]]; then
            echo "  Re-evaluating $(basename "$rep_dir")..."
            evaluate_rep "$rep_dir" "$task"
        fi
    done
}

# ── Experiment Tiers ──────────────────────────────────────────────────────

run_tier1() {
    echo ""
    echo "=== Tier 1: Model Comparison ==="
    echo "Models: ${MODELS[*]}"
    echo "Reps per model: $REPS"
    echo ""

    for model in "${MODELS[@]}"; do
        for ((r=1; r<=REPS; r++)); do
            echo "--- $model (rep $r/$REPS) ---"
            run_single "$TASK" "$model" "$FRAME_SKIP" "true"
        done
    done
}

run_tier2() {
    echo ""
    echo "=== Tier 2: Split Model Experiments ==="
    echo ""

    # Cheap intent, expensive decision
    echo "--- Intent: gemini-2.5-flash, Decision: gemini-2.5-pro ---"
    run_single "$TASK" "gemini-2.5-flash" "$FRAME_SKIP" "true" \
        "--decision-model gemini-2.5-pro --decision-backend gemini"

    # Expensive intent, cheap decision
    echo "--- Intent: gemini-2.5-pro, Decision: gemini-2.5-flash ---"
    run_single "$TASK" "gemini-2.5-pro" "$FRAME_SKIP" "true" \
        "--decision-model gemini-2.5-flash --decision-backend gemini"
}

run_tier3() {
    echo ""
    echo "=== Tier 3: Frame Skip Ablation ==="
    local best_model="gemini-2.5-pro"
    echo "Model: $best_model"
    echo ""

    for fs in 15 30 60 90; do
        echo "--- frame_skip=$fs ---"
        run_single "$TASK" "$best_model" "$fs" "true"
    done
}

run_tier4() {
    echo ""
    echo "=== Tier 4: Ground Truth Ablation ==="
    local best_model="gemini-2.5-pro"
    echo ""

    echo "--- With GT robot status ---"
    run_single "$TASK" "$best_model" "$FRAME_SKIP" "true"

    echo "--- Without GT robot status ---"
    run_single "$TASK" "$best_model" "$FRAME_SKIP" "false"
}

# ── Main ──────────────────────────────────────────────────────────────────

echo "============================================="
echo "  AURA Experiment Runner"
echo "  Task: $TASK"
echo "  Video: $VIDEO"
echo "  Experiments dir: $EXPERIMENTS_DIR"
echo "============================================="

case "$TIER" in
    1)   run_tier1 ;;
    2)   run_tier2 ;;
    3)   run_tier3 ;;
    4)   run_tier4 ;;
    all)
        run_tier1
        run_tier2
        run_tier3
        run_tier4
        ;;
    *)
        echo "Unknown tier: $TIER (valid: 1, 2, 3, 4, all)"
        exit 1
        ;;
esac

echo ""
echo "============================================="
echo "  All experiments complete!"
echo "  Results in: $EXPERIMENTS_DIR/"
echo "============================================="

# Aggregate results
echo ""
echo "Aggregating results..."
$PYTHON scripts/eval/aggregate_results.py --experiments-dir "$EXPERIMENTS_DIR" --output "$EXPERIMENTS_DIR/aggregate_results.json" 2>&1 || echo "[WARN] Aggregation failed — run aggregate_results.py manually"
