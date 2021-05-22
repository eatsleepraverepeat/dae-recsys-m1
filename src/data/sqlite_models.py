import tqdm
import sqlalchemy as sa
from sqlalchemy import Column, inspect
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()


class Playcounts(Base):
    __tablename__ = 'playcounts'

    index = Column(sa.Integer, unique=True, primary_key=True, nullable=False)
    user_id = Column(sa.Text, unique=False, primary_key=False, nullable=False)
    song_id = Column(sa.Text, unique=False, primary_key=False, nullable=False)
    playcount = Column(sa.Integer, unique=False, primary_key=False, nullable=False)

    def __init__(self, **kwargs):
        super(Playcounts, self).__init__(**kwargs)

    def __repr__(self):
        return f"<Playcount(user_id={self.user_id}, song_id={self.song_id}, playcount={self.playcount})>"


class Songs(Base):
    __tablename__ = 'songs'

    track_id = Column(sa.Text, unique=True, primary_key=True, nullable=False)
    title = Column(sa.Text)
    song_id = Column(sa.Text)
    release = Column(sa.Text)
    artist_id = Column(sa.Text)
    artist_mbid = Column(sa.Text)
    artist_name = Column(sa.Text)
    duration = Column(sa.REAL)
    artist_familiarity = Column(sa.REAL)
    artist_hotttnesss = Column(sa.REAL)
    year = Column(sa.Integer)
    track_7digitalid = Column(sa.Integer)
    shs_perf = Column(sa.Integer)
    shs_work = Column(sa.Integer)

    def as_dict(self):
        return {c.name: getattr(self, c.name) for c in self.__table__.columns}

    def __repr__(self):
        return f"" \
               f"<Song(" \
               f"track_id={self.track_id}, " \
               f"title={self.title}, " \
               f"song_id={self.song_id}, " \
               f"release={self.release}, " \
               f"artist_id={self.artist_id}, " \
               f"artist_name={self.artist_name} " \
               f"artist_familiarity={self.artist_familiarity} " \
               f"artist_hotttnesss={self.artist_hotttnesss}" \
               f">"
