#!/bin/bash

GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
BOLD='\033[1m'
NC='\033[0m'

WIDTH=60
BORDER=$(printf '#%.0s' $(seq 1 $WIDTH))

print_banner() {
    local color=$1
    local msg="$2"

    local padding=$(( (WIDTH - 2 - ${#msg}) / 2 ))
    local extra=$(( (WIDTH - 2 - ${#msg}) % 2 ))

    printf "\n${color}${BOLD}%s\n" "$BORDER"
    printf "#%*s%s%*s#\n" \
        $padding "" \
        "$msg" \
        $((padding + extra)) ""
    printf "%s${NC}\n\n" "$BORDER"
}