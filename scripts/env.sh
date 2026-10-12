# Source inside any job step. Activates $HARNESS_ENV (a venv or conda env path) if set,
# then puts the repo on PYTHONPATH.
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [ -n "$HARNESS_ENV" ]; then
  if [ -f "$HARNESS_ENV/bin/activate" ]; then source "$HARNESS_ENV/bin/activate"; else conda activate "$HARNESS_ENV"; fi
fi
export PYTHONPATH="$REPO" TOKENIZERS_PARALLELISM=false
cd "$REPO"
