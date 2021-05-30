FROM python:3.7
# Dependencies for librosa
RUN apt-get update -y && apt-get install -y --no-install-recommends build-essential gcc \
                                        libsndfile1 \
                                        ffmpeg

# Create a working directory.
RUN mkdir /app
WORKDIR /app

# Install python dependencies
COPY requirements.txt /
RUN pip install -r /requirements.txt

# Map a port
EXPOSE 8888 5000
CMD ["jupyter", "lab", "--ip='0.0.0.0'", "--port=8888", "--no-browser", "--allow-root"]
