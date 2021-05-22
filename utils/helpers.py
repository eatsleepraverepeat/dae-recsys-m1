from sqlalchemy import create_engine
from sqlalchemy import func as sa_func
from sqlalchemy.orm.session import sessionmaker
from src.data.sqlite_models import Playcounts, Songs


class MSDDataHelper:

    def __init__(self):
        self.db_uri = "sqlite:///data/track_metadata.db"
        self.engine = create_engine(self.db_uri)
        self.Session = sessionmaker()
        self.Session.configure(bind=self.engine)
        self.session = self.Session()

    def get_data(self, item_type: str = 'artist'):

        if item_type == 'artist':
            return self.session. \
                query(Playcounts.user_id, Songs.artist_id, sa_func.sum(Playcounts.playcount)). \
                join(Songs, Songs.song_id == Playcounts.song_id). \
                group_by(Playcounts.user_id, Songs.artist_id). \
                all()
