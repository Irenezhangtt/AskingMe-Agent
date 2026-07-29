#!/usr/bin/env bash
set -euo pipefail

COMPOSE_FILE="${COMPOSE_FILE:-docker-compose.yml}"

info() { printf '[INFO] %s\n' "$*"; }
warn() { printf '[WARN] %s\n' "$*" >&2; }
fail() { printf '[ERROR] %s\n' "$*" >&2; exit 1; }

check_dependencies() {
    command -v docker >/dev/null 2>&1 || fail "Docker is not installed."
    docker compose version >/dev/null 2>&1 || fail "Docker Compose is not installed."
}

check_environment() {
    if [[ ! -f .env ]]; then
        [[ -f .env.example ]] || fail ".env.example does not exist."
        cp .env.example .env
        chmod 600 .env
        warn "Created .env from .env.example. Configure ANTHROPIC_API_KEY and other secrets before starting."
        exit 1
    fi
}

create_directories() {
    mkdir -p data/chroma data/redis logs backups
}

start() {
    check_dependencies
    check_environment
    create_directories
    docker compose -f "$COMPOSE_FILE" up -d
    docker compose -f "$COMPOSE_FILE" ps
}

stop() {
    docker compose -f "$COMPOSE_FILE" down
}

restart() {
    docker compose -f "$COMPOSE_FILE" restart
    docker compose -f "$COMPOSE_FILE" ps
}

build() {
    check_dependencies
    docker compose -f "$COMPOSE_FILE" build
}

health() {
    docker compose -f "$COMPOSE_FILE" ps
    curl --fail --silent http://127.0.0.1/health >/dev/null \
        && info "Application health check passed." \
        || fail "Application health check failed."
}

backup() {
    local target="backups/$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$target"
    [[ -d data ]] && cp -R data "$target/"
    info "Backup created at $target"
}

restore() {
    local source="${1:-}"
    [[ -n "$source" ]] || fail "Provide a backup directory."
    [[ -d "$source/data" ]] || fail "Backup data directory does not exist: $source/data"
    warn "Restoring this backup will overwrite current local data."
    read -r -p "Continue? (y/N): " answer
    [[ "$answer" =~ ^[Yy]$ ]] || { info "Restore canceled."; return; }
    stop
    cp -R "$source/data/." data/
    start
}

cleanup() {
    warn "This removes containers and Compose-managed data volumes."
    read -r -p "Continue? (y/N): " answer
    [[ "$answer" =~ ^[Yy]$ ]] || { info "Cleanup canceled."; return; }
    docker compose -f "$COMPOSE_FILE" down --volumes --remove-orphans
}

usage() {
    cat <<'HELP'
AskingMe Agent Docker deployment utility

Usage: ./docker-deploy.sh COMMAND [ARGUMENT]

Commands:
  install          Check dependencies, create directories, and build images
  start            Start all services
  stop             Stop all services
  restart          Restart all services
  status           Show service status
  logs [SERVICE]   Follow logs for all services or one service
  health           Run the public health check
  build            Rebuild images
  backup           Back up local data
  restore PATH     Restore a backup
  cleanup          Remove containers and managed volumes
  help             Show this help
HELP
}

command_name="${1:-help}"
case "$command_name" in
    install) check_dependencies; check_environment; create_directories; build ;;
    start) start ;;
    stop) stop ;;
    restart) restart ;;
    status) docker compose -f "$COMPOSE_FILE" ps ;;
    logs)
        shift
        if [[ $# -gt 0 ]]; then
            docker compose -f "$COMPOSE_FILE" logs -f "$1"
        else
            docker compose -f "$COMPOSE_FILE" logs -f
        fi
        ;;
    health) health ;;
    build) build ;;
    backup) backup ;;
    restore) restore "${2:-}" ;;
    cleanup) cleanup ;;
    help|-h|--help) usage ;;
    *) fail "Unknown command: $command_name" ;;
esac
