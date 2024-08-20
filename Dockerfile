FROM python:3.10-slim

# Set the working directory
WORKDIR /app

# Copy the poetry lock file and pyproject.toml
COPY pyproject.toml poetry.lock ./

# Install poetry
RUN pip install poetry
RUN poetry config virtualenvs.in-project true

# Install the dependencies
RUN poetry install --no-root

# Copy the Gradio app code
COPY . .

# Expose the port Gradio will run on
EXPOSE 7860
