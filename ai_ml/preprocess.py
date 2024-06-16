import cv2
import torch
import librosa
import numpy as np
from transformers import VideoMAEImageProcessor, VideoMAEModel, Wav2Vec2Processor, Wav2Vec2Model
import config
from database import VideoFingerprint
from sqlalchemy import func

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load pre-trained models with warning suppression for uninitialized weights
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    video_processor = VideoMAEImageProcessor.from_pretrained("MCG-NJU/videomae-base")
    video_model = VideoMAEModel.from_pretrained("MCG-NJU/videomae-base").to(device)
    audio_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    audio_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h").to(device)

def extract_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    chunk_size = config.VIDEO_CHUNK_DURATION * fps
    frames = []
    chunk_start = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (224, 224))  # Ensure frames are resized to the expected input size
        frames.append(frame)
        if len(frames) == chunk_size:
            frames = np.array(frames)  # Convert list to numpy array for faster tensor conversion
            frames = frames.transpose((0, 3, 1, 2))  # Ensure frames are in the correct shape (B, H, W, C) to (B, C, H, W)
            yield frames, chunk_start // fps, (chunk_start + chunk_size) // fps
            frames = []
        chunk_start += 1
    cap.release()

def extract_audio(video_path):
    try:
        audio, sr = librosa.load(video_path, sr=16000)
    except Exception as e:
        print(f"Error loading audio: {e}")
        # Use an alternative method if librosa fails
        import subprocess
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.wav') as temp_wav:
            subprocess.run(['ffmpeg', '-i', video_path, '-ar', '16000', '-ac', '1', temp_wav.name], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            audio, sr = librosa.load(temp_wav.name, sr=16000)
    return audio, sr

def generate_video_fingerprint(frames):
    inputs = video_processor(frames.tolist(), return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = video_model(**inputs)
    
    return outputs.last_hidden_state.mean(dim=1).cpu().numpy()

def generate_audio_fingerprint(audio, start_time, end_time):
    inputs = audio_processor(audio[start_time:end_time], return_tensors="pt", sampling_rate=16000)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = audio_model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).cpu().numpy()

def match_fingerprints(new_fingerprint_video, new_fingerprint_audio, session, threshold=config.SIMILARITY_THRESHOLD):
    video_similarity_query = session.query(
        VideoFingerprint,
        (1 - func.cosine_distance(VideoFingerprint.fingerprint_video, new_fingerprint_video)).label('similarity')
    ).order_by(func.desc('similarity')).limit(1).first()
    
    audio_similarity_query = session.query(
        VideoFingerprint,
        (1 - func.cosine_distance(VideoFingerprint.fingerprint_audio, new_fingerprint_audio)).label('similarity')
    ).order_by(func.desc('similarity')).limit(1).first()
    
    max_similarity_query = max(video_similarity_query, audio_similarity_query, key=lambda x: x.similarity) if video_similarity_query and audio_similarity_query else video_similarity_query or audio_similarity_query
    
    if max_similarity_query and max_similarity_query.similarity >= threshold:
        return {
            'video_name': max_similarity_query.VideoFingerprint.video_name,
            'start_time': max_similarity_query.VideoFingerprint.start_time,
            'end_time': max_similarity_query.VideoFingerprint.end_time,
            'similarity': max_similarity_query.similarity
        }
    return None

def consolidate_results(results):
    consolidated = []
    prev = None

    for result in results:
        if prev is None:
            prev = result
            continue

        if (prev['license_video'] == result['license_video'] and
            int(result['source_interval'].split('-')[0]) == int(prev['source_interval'].split('-')[1]) and
            int(result['license_interval'].split('-')[0]) == int(prev['license_interval'].split('-')[1])):
            prev['source_interval'] = f"{prev['source_interval'].split('-')[0]}-{result['source_interval'].split('-')[1]}"
            prev['license_interval'] = f"{prev['license_interval'].split('-')[0]}-{result['license_interval'].split('-')[1]}"
        else:
            consolidated.append(prev)
            prev = result
    if prev is not None:
        consolidated.append(prev)
    return consolidated
