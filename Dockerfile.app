# ==============================================================================
# Universal-Node-Resolver — Gradio UI / Client Container
# ==============================================================================
# This Dockerfile packages the interactive split-screen UI for deployment 
# to a Hugging Face Space. It acts as the "Agent Client" making inference
# calls and updating the environment state.

FROM python:3.10-slim

# Prevent Python from writing pyc files and keep stdout unbuffered
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install Gradio and the required OpenEnv client libraries
RUN pip install --no-cache-dir gradio openenv pydantic

# Copy the entire project context (Client agent, models, and UI)
# In production, this client queries the deployed RL server.
COPY . .

# Gradio's standard port
EXPOSE 7860

# Launch the interactive split-screen demo
CMD ["python", "app.py"]
