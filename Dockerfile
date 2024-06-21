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

# Copy only the necessary files from the build stage
COPY --from=builder /usr/local/lib/python3.8/site-packages /usr/local/lib/python3.8/site-packages
COPY --from=builder /app/clearml_hf_pipeline.py .
COPY --from=builder /app/custom_dataset.csv .

# Run the script
CMD ["python", "clearml_hf_pipeline.py"]
