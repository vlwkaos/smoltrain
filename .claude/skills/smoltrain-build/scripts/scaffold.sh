#!/usr/bin/env bash
# scaffold.sh — Create a new smoltrain task directory with taxonomy.yaml
# Usage: scaffold.sh <work_dir> <task_name> <goal> <n_per_class> <languages>
#        Then write classes via stdin as YAML block (passed as heredoc)
# All class descriptions must be passed via --classes-yaml <file>
set -euo pipefail

WORK_DIR="$1"
TASK_NAME="$2"
GOAL="$3"
N_PER_CLASS="${4:-500}"
LANGUAGES="${5:-en}"   # "en" or "en,ko" or "en,ko,mixed"
CLASSES_YAML="$6"      # path to a YAML file with classes block

mkdir -p "$WORK_DIR/data" "$WORK_DIR/models"

# Build languages list
LANG_LIST=""
IFS=',' read -ra LANGS <<< "$LANGUAGES"
for lang in "${LANGS[@]}"; do
  LANG_LIST+="  - ${lang}"$'\n'
done

# taxonomy.yaml
cat > "$WORK_DIR/taxonomy.yaml" <<TAXEOF
classes:
$(cat "$CLASSES_YAML")

languages:
${LANG_LIST}
config:
  samples_per_class: ${N_PER_CLASS}
  naturalizer_ratio: 0.20
  hard_negative_ratio: 0.10
  latency_target_ms: 50
  f1_floor: 0.85
  agentic_recall_floor: 0.87
TAXEOF

echo "Scaffolded: $WORK_DIR/taxonomy.yaml"
echo "Task: $TASK_NAME  |  Goal: $GOAL  |  n_per_class: $N_PER_CLASS  |  langs: $LANGUAGES"
