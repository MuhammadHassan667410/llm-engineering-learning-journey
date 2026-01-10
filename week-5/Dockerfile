# Use an official lightweight Python image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file first to leverage Docker cache
COPY requirements.txt .

# Install dependencies
# (We install build-essential just in case some packages need compilation)
RUN apt-get update && apt-get install -y build-essential && \
    pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY improved_app.py .
COPY implementation/ implementation/

# Copy the Vector Database (The "Brain")
# Note: This increases image size. For production, consider using an external vector DB or mounting a volume.
COPY vector_db/ vector_db/

# Expose the port Gradio runs on
EXPOSE 7860

# Set environment variables
# GRADIO_SERVER_NAME="0.0.0.0" is crucial for Docker networking
ENV GRADIO_SERVER_NAME="0.0.0.0"
ENV GRADIO_SERVER_PORT="7860"

# Command to run the app
CMD ["python", "improved_app.py"]
