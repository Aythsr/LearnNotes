#!/usr/bin/env bash

set -euo pipefail

if command -v uv >/dev/null 2>&1; then
	echo "uv is already installed"
else
	# Install uv using the official installer when the command is unavailable.
    echo "Installing uv..."
	pip install uv
	# wget -qO- https://astral.sh/uv/install.sh | sh
fi

