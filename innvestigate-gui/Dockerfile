# Base image
FROM python:3.9

# Set the working directory
WORKDIR /innvestigate-gui

# Copy the backend code
COPY . .

# Install dependencies
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
RUN ["pip", "install", "-r", "requirements.txt"]

# Expose the port
EXPOSE 5000

RUN cp -f flask_uploads.py /usr/local/lib/python3.9/site-packages/flask_uploads.py

ENV FLASK_APP=innvestigate-gui

# Set the entry point
CMD [ "python3", "-m" , "flask", "run", "--host=0.0.0.0"]