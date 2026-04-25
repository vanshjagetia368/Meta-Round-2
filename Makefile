# ==============================================================================
# Universal-Node-Resolver — Hackathon Deployment Makefile
# ==============================================================================

.PHONY: install test run-env run-ui push-space

# Installs all necessary dependencies for local development and training
install:
	pip install gradio openenv pydantic pytest matplotlib
	@echo "Note: unsloth and trl should be installed on your GPU instance manually."

# Runs the rigorous QA Pytest suite to verify the Anti-Cheat Nuke logic
test:
	pytest tests/ -v

# Starts the local OpenEnv server for manual API testing
run-env:
	openenv serve openenv.yaml --host 0.0.0.0 --port 8000

# Starts the Gradio interactive split-screen UI locally
run-ui:
	python app.py

# Packages and deploys the UI to a Hugging Face Space
push-space:
	@echo "Uploading Gradio Demo to Hugging Face Spaces..."
	huggingface-cli upload space-name/universal-node-resolver . --repo-type=space
