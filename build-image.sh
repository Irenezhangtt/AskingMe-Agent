#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME="${IMAGE_NAME:-askingme}"
REGISTRY="${REGISTRY:-}"
VERSION="${VERSION:-latest}"
NO_CACHE=false
PLATFORMS=""

info() { printf '[INFO] %s\n' "$*"; }
fail() { printf '[ERROR] %s\n' "$*" >&2; exit 1; }

usage() {
    cat <<'HELP'
AskingMe Agent Docker image build utility

Usage: ./build-image.sh COMMAND [OPTIONS]

Commands:
  build         Build the default image
  build-prod    Build the production target
  build-dev     Build the development target
  build-test    Build the test target
  push          Push version and latest tags
  tag VERSION   Add a version tag
  clean         Remove Docker build cache
  help          Show this help

Options:
  --no-cache
  --platform PLATFORM
  --registry REGISTRY
  --version VERSION
HELP
}

parse_options() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --no-cache) NO_CACHE=true; shift ;;
            --platform) PLATFORMS="$2"; shift 2 ;;
            --registry) REGISTRY="${2%/}/"; shift 2 ;;
            --version) VERSION="$2"; shift 2 ;;
            *) fail "Unknown option: $1" ;;
        esac
    done
}

build_image() {
    local target="$1"
    shift
    parse_options "$@"
    local args=(build)
    [[ -n "$target" ]] && args+=(--target "$target")
    [[ "$NO_CACHE" == true ]] && args+=(--no-cache)
    [[ -n "$PLATFORMS" ]] && args+=(--platform "$PLATFORMS")
    local full_tag="${REGISTRY}${IMAGE_NAME}:${VERSION}"
    args+=(-t "$full_tag" -t "${REGISTRY}${IMAGE_NAME}:latest" .)
    docker "${args[@]}"
    info "Image built: $full_tag"
}

command_name="${1:-help}"
shift || true
case "$command_name" in
    build) build_image "" "$@" ;;
    build-prod) build_image production "$@" ;;
    build-dev) build_image development "$@" ;;
    build-test) build_image test "$@" ;;
    push)
        parse_options "$@"
        docker push "${REGISTRY}${IMAGE_NAME}:${VERSION}"
        docker push "${REGISTRY}${IMAGE_NAME}:latest"
        ;;
    tag)
        new_version="${1:-}"
        [[ -n "$new_version" ]] || fail "Provide a new version tag."
        docker tag "${REGISTRY}${IMAGE_NAME}:${VERSION}" "${REGISTRY}${IMAGE_NAME}:${new_version}"
        ;;
    clean) docker builder prune -f ;;
    help|-h|--help) usage ;;
    *) fail "Unknown command: $command_name" ;;
esac
