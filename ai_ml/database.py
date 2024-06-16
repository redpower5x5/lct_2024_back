from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import sessionmaker
from pgvector.sqlalchemy import Vector
import config

Base = declarative_base()
engine = create_engine(config.DATABASE_URL)
Session = sessionmaker(bind=engine)
session = Session()

class VideoFingerprint(Base):
    __tablename__ = 'video_fingerprints'
    
    id = Column(Integer, primary_key=True)
    video_name = Column(String, nullable=False)
    start_time = Column(Integer, nullable=False)
    end_time = Column(Integer, nullable=False)
    fingerprint_audio = Column(Vector(768), nullable=False)
    fingerprint_video = Column(Vector(768), nullable=False)

def create_tables():
    Base.metadata.create_all(engine)

if __name__ == "__main__":
    create_tables()
