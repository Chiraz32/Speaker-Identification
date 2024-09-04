FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Clear pip cache to avoid any corrupted downloads
RUN pip cache purge

# Copy the current directory contents into the container at /app
COPY . /app

# Copy requirements.txt into the container
COPY requirements.txt /app/

RUN pip cache purge
# Upgrade pip and setuptools
RUN pip install --no-cache-dir --upgrade pip setuptools

# Install basic packages
RUN pip install --no-cache-dir fastapi uvicorn

# Install large packages separately with no dependencies flag
RUN pip install --no-cache-dir --no-deps torch numpy
RUN pip install --no-cache-dir pyannote.audio soundfile

# Install remaining packages from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port that the FastAPI app will run on
EXPOSE 8000

# Run the FastAPI app with Uvicorn
CMD ["uvicorn", "model_api:app", "--host", "0.0.0.0", "--port", "8000"]

