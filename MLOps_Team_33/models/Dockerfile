# Start with a base Python image
FROM python:3.8-slim

# Set the working directory in the container
WORKDIR /app

# Copy your requirements file to the container
COPY requirements.txt requirements.txt

# Add this line to copy the script to the container
COPY your_model_script.py your_model_script.py


# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy your MLflow artifacts (or other model files) into the container
COPY model.pkl model.pkl
COPY conda.yaml conda.yaml
COPY python_env.yaml python_env.yaml

# Expose a port (if necessary, e.g., for serving your model via an API)
EXPOSE 5000

# Set the default command for the container
CMD ["python", "your_model_script.py"]
