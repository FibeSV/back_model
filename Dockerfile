# Install base Python image
FROM python:3.8.10

# Copy files to the container
COPY server.py /app/
COPY requirements.txt /app/
COPY model.h5 /app/
COPY model.joblib /app/
COPY modelim.py /app/
COPY config.json /app/
# Set working directory to previously added app directory
WORKDIR /app/

# Install dependencies
RUN pip install -r requirements.txt


# Expose the port uvicorn is running on
EXPOSE 80

# Run uvicorn server
CMD ["uvicorn", "server:app", "--reload", "--host", "0.0.0.0", "--port", "80"]