#!/usr/bin/env bash
# Copies the agent into a microduck_rl checkout.
# Usage: ./install.sh path/to/microduck_rl
set -e

RL_DIR="${1:-$HOME/microduck_rl}"
HERE="$(cd "$(dirname "$0")" && pwd)"
SCENE_DIR="$RL_DIR/src/mjlab_microduck/robot/microduck"

if [ ! -f "$RL_DIR/scripts/infer_policy.py" ]; then
    echo "Can't find microduck_rl at $RL_DIR"
    echo "Usage: ./install.sh path/to/microduck_rl"
    exit 1
fi

cp "$HERE"/agent/*.py "$RL_DIR/"
cp "$HERE/scenes/scene_eyes.xml" "$SCENE_DIR/"

echo "Copied agent scripts to $RL_DIR"
echo "Copied scene_eyes.xml to $SCENE_DIR"

if [ ! -f "$RL_DIR/pretrained/velstand.onnx" ]; then
    echo
    echo "Note: pretrained/velstand.onnx not found. Download the policies with:"
    echo "  cd $RL_DIR && uvx hf download pollen-robotics/microduck-policies --local-dir pretrained"
fi
