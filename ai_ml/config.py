import os

DATABASE_URL = "postgresql://postgres:postgres@localhost:5432/videodb"
LICENSED_VIDEO_PATH = "data/licensed_videos/"
TRAIN_VIDEO_PATH = "data/train_videos/"
VIDEO_CHUNK_DURATION = 15  # Duration of video chunks in seconds
SIMILARITY_THRESHOLD = 0.8  # Threshold for cosine similarity
