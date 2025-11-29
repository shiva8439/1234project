# Base Python 3.11 image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy files
COPY . .

# Install dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Expose port
EXPOSE 5000

# Run the app
CMD ["gunicorn", "4:app", "--bind", "0.0.0.0:5000"]
