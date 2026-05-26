#!/bin/sh
set -e
case "$1" in
  api)     exec uvicorn api.asgi:app --host 0.0.0.0 --port 8000 ;;
  worker)  exec python -m worker.main ;;
  alembic) shift; exec alembic -c /app/packages/common/db/alembic.ini "$@" ;;
  *)       exec "$@" ;;
esac
