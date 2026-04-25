# ==============================================================================
# Universal-Node-Resolver — OpenEnv Server Container
# ==============================================================================
# This Dockerfile isolates the Reinforcement Learning environment from the
# agent client, strictly adhering to the OpenEnv client/server separation.
# It packages the UniversalMockRegistry and exposes the environment via HTTP.

FROM python:3.10-slim

# Prevent Python from writing pyc files and keep stdout unbuffered
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install standard core dependencies for the environment
RUN pip install --no-cache-dir openenv pydantic

# Copy only the server-side logic and the OpenEnv manifest
# The agent client code is intentionally excluded from this container
COPY server/ ./server/
COPY openenv.yaml .

# OpenEnv standard port
EXPOSE 8000

# Launch the OpenEnv server using the standard CLI, binding to all interfaces
CMD ["openenv", "serve", "openenv.yaml", "--host", "0.0.0.0", "--port", "8000"]
