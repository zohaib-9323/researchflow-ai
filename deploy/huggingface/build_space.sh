#!/bin/sh
# Copy the files a Hugging Face Space needs into a local clone of the Space repository.
#   deploy/huggingface/build_space.sh ../researchflow-ai-space
set -e
TARGET=${1:?usage: build_space.sh <path-to-cloned-space-repo>}
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
[ -d "$TARGET/.git" ] || { echo "$TARGET is not a git clone of your Space"; exit 1; }

for dir in app frontend evaluation .streamlit deploy/huggingface; do
  mkdir -p "$TARGET/$dir"
  rsync -a --delete --exclude '__pycache__' --exclude 'results' --exclude '.env*' \
    "$ROOT/$dir/" "$TARGET/$dir/"
done
cp "$ROOT/pyproject.toml" "$TARGET/"
cp "$ROOT/deploy/huggingface/Dockerfile" "$TARGET/Dockerfile"
cp "$ROOT/deploy/huggingface/README.md" "$TARGET/README.md"
echo "Space files ready in $TARGET — review with 'git status', then commit and push."
