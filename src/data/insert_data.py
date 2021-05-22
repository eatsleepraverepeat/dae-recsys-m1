import tqdm
import zipfile
from pathlib import Path
import more_itertools as mit
from sqlalchemy import create_engine, inspect
from sqlalchemy.orm.session import sessionmaker
from src.data.sqlite_models import Playcounts, Songs


FILENAME_PATH = Path('../../data/train_triplets.txt')
MSD_RAW_TRIPLETS_PATH = Path('../../data/train_triplets.txt.zip')

MSD_METADB_URI = "sqlite:///data/track_metadata.db"
TRIPLETS_DATA_PATH = Path('../../data/train_triplets.txt')
CHUNK_SIZE = 10000

engine = create_engine(MSD_METADB_URI)
Inspector = inspect(engine)

if not Inspector.has_table(table_name=Playcounts.__tablename__):
    Playcounts.__table__.create(bind=engine)

Session = sessionmaker()
Session.configure(bind=engine)
session = Session()

with zipfile.ZipFile(MSD_RAW_TRIPLETS_PATH, mode='r') as f:
    f.extract(FILENAME_PATH.name, FILENAME_PATH.parts[0])

TOTAL_TRIPLETS = 48373586
GLOBAL_ITERATOR = 0
with TRIPLETS_DATA_PATH.open(mode='r') as f:
    for chunk in tqdm.tqdm(
        mit.chunked(f, CHUNK_SIZE),
        total=(TOTAL_TRIPLETS // CHUNK_SIZE) + 1
    ):

        objects = []
        for row in chunk:
            user, item, playcount = row.strip().split()
            objects.append(
                Playcounts(**{
                    "index": GLOBAL_ITERATOR,
                    "user_id": user,
                    "song_id": item,
                    "playcount": playcount
                })
            )
            GLOBAL_ITERATOR += 1

        session.bulk_save_objects(objects)
        session.commit()
