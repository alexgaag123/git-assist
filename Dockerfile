FROM python:3.12-slim AS build
WORKDIR /app
COPY pyproject.toml ./
COPY packages ./packages
COPY apps ./apps
RUN pip install --no-cache-dir '.[postgres]'

FROM python:3.12-slim AS runtime
WORKDIR /app
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates git \
    && rm -rf /var/lib/apt/lists/*
COPY --from=build /usr/local /usr/local
COPY packages ./packages
COPY apps ./apps
# alembic.ini lives in packages/common/db with `script_location = %(here)s/migrations`, so it
# only resolves in place; the `alembic` entrypoint case points `-c` at it (never copy it to /app).
ENV PYTHONPATH=/app/packages:/app/apps
COPY docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod +x /usr/local/bin/docker-entrypoint.sh
ENTRYPOINT ["docker-entrypoint.sh"]
CMD ["api"]
