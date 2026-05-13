#!/bin/bash
# Step 0: Bootstrap submission dataset repo
#
# Duplicates openai/gdpval into your HF submission repo (if not exists)
# then downloads a local snapshot to data/gdpval-local.
#
# submission_repo_id는 experiment YAML의 data.source에서 자동으로 읽습니다.
#
# Usage:
#   export HF_TOKEN=hf_xxx
#   ./step0_bootstrap.sh <yaml_config_path>
#   ./step0_bootstrap.sh experiments/exp001_smoke_baseline.yaml
#
# This is idempotent — safe to run multiple times.

set -euo pipefail
cd "$(dirname "$0")"

YAML_CONFIG="${1:?Usage: ./step0_bootstrap.sh <yaml_config_path>}"

if [ ! -f "$YAML_CONFIG" ]; then
  echo "❌ YAML config not found: $YAML_CONFIG"
  exit 1
fi

if [ -z "${HF_TOKEN:-}" ]; then
  echo "❌ HF_TOKEN not set."
  echo "   export HF_TOKEN=hf_xxx"
  exit 1
fi

export YAML_CONFIG

python3 - <<'PYEOF'
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) or ".")
from core.experiment_config import ExperimentConfig
from core.repo_bootstrapper import RepoBootstrapper

config = ExperimentConfig.from_yaml(os.environ["YAML_CONFIG"])
repo_id = config.data_filter.source
if not repo_id:
    raise ValueError("data.source not found in YAML")

expected_rows = config.data_filter.sample_size

print(f"ℹ️  Submission repo: {repo_id}  (from {os.environ['YAML_CONFIG']})")
print("🔧 Step 0: Bootstrap Submission Repo")
print(f"   Repo: {repo_id}")
if expected_rows is not None:
    print(f"   Expected rows: {expected_rows}")
print("")

bs = RepoBootstrapper(
    submission_repo_id=repo_id,
    expected_rows=expected_rows,
)
bs.bootstrap()
PYEOF
