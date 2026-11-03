#!/usr/bin/env bash
# Private Ollama, reranker and Transformers forwarding through an SSH relay.
set -u

if [[ $# -lt 2 ]]; then
  echo "Usage: bash $0 {tu|development} SSH_RELAY [ssh options, e.g. -i KEY]" >&2
  exit 2
fi
role=$1
relay=$2
shift 2

case "$role" in
  tu)
    forwards=(-R 127.0.0.1:11435:127.0.0.1:11434
              -R 127.0.0.1:18010:127.0.0.1:8010
              -R 127.0.0.1:15001:127.0.0.1:5001)
    ;;
  development)
    forwards=(-L 127.0.0.1:11434:127.0.0.1:11435
              -L 127.0.0.1:8010:127.0.0.1:18010
              -L 127.0.0.1:15001:127.0.0.1:15001)
    ;;
  *) echo "Unknown role: $role" >&2; exit 2 ;;
esac

ssh_pid=""
stop() {
  if [[ -n "$ssh_pid" ]]; then
    kill "$ssh_pid" 2>/dev/null || true
    wait "$ssh_pid" 2>/dev/null || true
  fi
  exit 0
}
trap stop INT TERM

while true; do
  echo "[$(date --iso-8601=seconds)] Starting $role model tunnel via $relay"
  ssh -N -T -S none \
    -o BatchMode=yes -o ExitOnForwardFailure=yes \
    -o ServerAliveInterval=30 -o ServerAliveCountMax=3 \
    -o StrictHostKeyChecking=yes -o ConnectTimeout=10 \
    "${forwards[@]}" "$@" "$relay" &
  ssh_pid=$!
  wait "$ssh_pid"
  rc=$?
  ssh_pid=""
  echo "Tunnel exited with code $rc; retrying in 10 seconds."
  sleep 10
done
