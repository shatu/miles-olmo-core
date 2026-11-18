#!/usr/bin/env bash
# Builds the OLMo-core runtime image (docker/olmo/Dockerfile) from this repository at HEAD.
# With --upload, it also uploads the image to Beaker as <account>/miles-olmo-<commit>.
#
# The build context is `git archive HEAD`, so the image holds exactly that commit's Miles.
# The base is Open Instruct's pinned binary base, Beaker image 01M5ZYQZVF7DDJ1VR619KRSS42
# (Docker ID sha256:947e615f...). It is pulled once into the local image store and tagged
# open-instruct-miles-binary-base:20261027.
#
#   bash docker/olmo/build_image.sh            # build localhost/miles-olmo:<commit>
#   bash docker/olmo/build_image.sh --upload   # build it, then upload it to Beaker
#
# Refuses uncommitted changes to tracked files (the image would silently leave them out),
# and an upload whose name already exists on Beaker. `beaker image pull/create` talk to a
# Docker API. With podman, run `podman system service --time=0 unix:///run/podman/podman.sock &`
# and set DOCKER_HOST=unix:///run/podman/podman.sock.
set -euo pipefail

workspace=ai2/oe-agents-holmes
budget=ai2/oe-omai
upload=0
base_beaker=01M5ZYQZVF7DDJ1VR619KRSS42
base_tag=open-instruct-miles-binary-base:20261027
while [ $# -gt 0 ]; do
    case "$1" in
        --upload) upload=1; shift ;;
        --workspace) workspace=$2; shift 2 ;;
        --budget) budget=$2; shift 2 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

repo=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
if [ -n "$(git -C "$repo" status --porcelain --untracked-files=no)" ]; then
    echo "uncommitted changes to tracked files; commit them first, the image is built from HEAD:" >&2
    git -C "$repo" status --short --untracked-files=no >&2
    exit 1
fi
commit=$(git -C "$repo" rev-parse HEAD)
name=miles-olmo-${commit:0:10}
tag=localhost/miles-olmo:${commit:0:10}

if [ "$upload" = 1 ]; then
    account=$(beaker account whoami --format json \
        | python3 -c "import json, sys; d = json.load(sys.stdin); print((d[0] if isinstance(d, list) else d)['name'])")
    if beaker image get "$account/$name" > /dev/null 2>&1; then
        echo "Beaker image $account/$name already exists" >&2
        exit 1
    fi
fi

if ! docker image inspect "$base_tag" > /dev/null 2>&1; then
    beaker image pull "$base_beaker" "$base_tag"
fi

context=$(mktemp -d)
trap 'rm -rf "$context"' EXIT
# The commit read above, not HEAD: the base pull can take long enough for HEAD to move.
git -C "$repo" archive "$commit" | tar -x -C "$context"

docker build "$context" -f "$context/docker/olmo/Dockerfile" -t "$tag" \
    --platform linux/amd64 --build-arg BASE_IMAGE="$base_tag" --label miles.commit="$commit"

if [ "$upload" = 1 ]; then
    beaker image create --name "$name" --workspace "$workspace" --budget "$budget" \
        --description "OLMo-core runtime (docker/olmo/Dockerfile) with Miles at $commit" "$tag"
    echo "image: beaker:$account/$name"
fi
