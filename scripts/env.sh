# Source inside any job step. Activates $PSC_ENV (a venv or conda env path) if set,
# then puts the repo on PYTHONPATH.
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [ -n "$PSC_ENV" ]; then
  if [ -f "$PSC_ENV/bin/activate" ]; then source "$PSC_ENV/bin/activate"; else conda activate "$PSC_ENV"; fi
fi
export PYTHONPATH="$REPO" TOKENIZERS_PARALLELISM=false
cd "$REPO"
