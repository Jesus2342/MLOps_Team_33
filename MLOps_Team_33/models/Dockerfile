
FROM python:3.8-slim


WORKDIR /app

# requirements file to the container
COPY requirements.txt requirements.txt

# renamed script to the container
COPY MLB_student.py MLB_student.py

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install other necessary files
COPY model.pkl model.pkl
COPY conda.yaml conda.yaml
COPY python_env.yaml python_env.yaml


EXPOSE 5000

# Set the default command for the container
CMD ["python", "MLB_student.py"]
