FROM python:3.9.15-slim-buster

# Install system dependencies
# RUN pip install --upgrade pip

ADD ./requirements.txt ./requirements.txt

RUN pip install -r requirements.txt

# Copy the app into the container
COPY . /app

# Set the working directory to /app
WORKDIR /app/app

# Expose port 8501
EXPOSE 8501

# Run the app
CMD ["streamlit", "run", "chatbot_app.py"]