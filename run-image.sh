#!/usr/bin/env bash
set -euo pipefail

IMAGE_TAG="${IMAGE_TAG:-askingme-agent:latest}"
CONTAINER_NAME="${CONTAINER_NAME:-askingme-agent}"
ENV_FILE="${ENV_FILE:-.env}"
API_PORT="${API_PORT:-8000}"

info() { printf '[INFO] %s\n' "$*"; }
warn() { printf '[WARN] %s\n' "$*" >&2; }
fail() { printf '[ERROR] %s\n' "$*" >&2; exit 1; }

usage() {
    cat <<'HELP'
AskingMe Agent Docker image utility

Usage: ./run-image.sh COMMAND

Commands:
  run        Start the production container
  run-dev    Start a development container
  run-test   Run the test suite in a temporary container
  stop       Stop and remove the container
  restart    Restart the container
  logs       Follow container logs
  shell      Open a shell in the running container
  status     Show container status
  clean      Remove the container and its anonymous volumes
  help       Show this help

Environment variables:
  IMAGE_TAG, CONTAINER_NAME, ENV_FILE, API_PORT
HELP
}

require_env() {
    [[ -f "$ENV_FILE" ]] || fail "Environment file does not exist: $ENV_FILE"
}

remove_existing() {
    if docker container inspect "$CONTAINER_NAME" >/dev/null 2>&1; then
        warn "Removing existing container: $CONTAINER_NAME"
        docker rm -f "$CONTAINER_NAME" >/dev/null
    fi
}

run_container() {
    local mode="${1:-production}"
    require_env
    remove_existing
    local args=(run --name "$CONTAINER_NAME" --env-file "$ENV_FILE" -p "${API_PORT}:8000")
    [[ "$mode" == "production" ]] && args+=(-d --restart unless-stopped)
    args+=("$IMAGE_TAG")
    [[ "$mode" == "development" ]] && args+=(python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload)
    docker "${args[@]}"
    info "Container started. API: http://localhost:${API_PORT}"
}

command_name="${1:-help}"
case "$command_name" in
    run) run_container production ;;
    run-dev) run_container development ;;
    run-test) require_env; docker run --rm --env-file "$ENV_FILE" "$IMAGE_TAG" python -m unittest discover -s tests -v ;;
    stop) docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || warn "Container is not running." ;;
    restart) docker restart "$CONTAINER_NAME" ;;
    logs) docker logs -f "$CONTAINER_NAME" ;;
    shell) docker exec -it "$CONTAINER_NAME" /bin/sh ;;
    status) docker ps -a --filter "name=^/${CONTAINER_NAME}$" ;;
    clean) docker rm -f -v "$CONTAINER_NAME" >/dev/null 2>&1 || true ;;
    help|-h|--help) usage ;;
    *) fail "Unknown command: $command_name" ;;
esac
