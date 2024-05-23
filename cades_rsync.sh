#!/usr/bin/env bash

set -eox pipefail

# sync code
rsync -avz -e "ssh -A -i ~/.ssh/id_ed25519" \
	--exclude __pycache__ --exclude .mypy_cache --exclude .venv --exclude .pytest_cache --exclude .ruff_cache \
	--exclude .devenv --exclude .direnv --exclude .git \
	--exclude ./data \
	../trait-prediction \
	cloud-cades:/datax/dkishore/

# sync data
rsync -avz -e "ssh -A -i ~/.ssh/id_ed25519" \
	data \
	cloud-cades:/datax/dkishore/trait-prediction/
