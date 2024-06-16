import os
import pandas as pd
from sqlalchemy.orm import sessionmaker
from database import create_engine, Base
import config
from utils import extract_audio, extract_frames, generate_video_fingerprint, generate_audio_fingerprint, match_fingerprints, consolidate_results

def process_video_chunks(video_path, video_name, session):
    audio = extract_audio(video_path)
    results = []
    for frames, start_time, end_time in extract_frames(video_path):
        video_fingerprint = generate_video_fingerprint(frames)
        audio_fingerprint = generate_audio_fingerprint(audio, start_time * 16000, end_time * 16000)

        match = match_fingerprints(video_fingerprint, audio_fingerprint, session)

        if match:
            results.append({
                'source_video': video_name,
                'source_interval': f"{start_time}-{end_time}",
                'license_video': match['video_name'],
                'license_interval': f"{match['start_time']}-{match['end_time']}",
                'similarity': match['similarity']
            })
    return results

def main():
    engine = create_engine(config.DATABASE_URL)
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()

    test_videos = os.listdir(config.TRAIN_VIDEO_PATH)  # Change this to test video path as needed
    all_results = []

    for video_file in test_videos:
        video_path = os.path.join(config.TRAIN_VIDEO_PATH, video_file)
        results = process_video_chunks(video_path, video_file, session)
        all_results.extend(results)
    
    consolidated_results = consolidate_results(all_results)
    df = pd.DataFrame(consolidated_results)
    print(df)

if __name__ == '__main__':
    main()
