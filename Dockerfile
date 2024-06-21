# Stage 1: Build Stage
FROM python:3.8-slim as builder

# Set the working directory
WORKDIR /app

# Copy the requirements file
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the script and dataset
COPY clearml_hf_pipeline.py .
COPY custom_dataset.csv .

# Stage 2: Runtime Stage
FROM python:3.8-slim

# Set the working directory
WORKDIR /app

# Install Git
RUN apt-get update && apt-get install -y git

# Copy only the necessary files from the build stage
COPY --from=builder /usr/local/lib/python3.8/site-packages /usr/local/lib/python3.8/site-packages
COPY --from=builder /app/clearml_hf_pipeline.py .
COPY --from=builder /app/custom_dataset.csv .
COPY --from=builder /app/requirements.txt .

# Initialize Git repository and add remote origin
RUN git init
RUN git remote add origin https://github.com/ido83/clearml-demo-minikube-pipeline-hf.git

# Run the script
CMD ["python", "clearml_hf_pipeline.py"]
