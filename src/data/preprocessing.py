import json
import pandas as pd
from pathlib import Path

import numpy as np
from typing import List, Tuple
from scipy import sparse as sp

from utils.helpers import MSDDataHelper


MSD_SPLITS_DATA_PATH = Path('../../data/splits')
MSD_CSV_TRIPLETS_PATH = Path('../../data/msd-triplets_artists.csv')


def save_npz(filename: str, sparse_matrix: sp.csc_matrix):
    sp.save_npz(
        file=MSD_SPLITS_DATA_PATH.joinpath(filename),
        matrix=sparse_matrix
    )


def get_count(df: pd.DataFrame, column: str) -> pd.DataFrame:
    playback_count = df[[column]].groupby(column, as_index=False)
    return playback_count.size()


def filter_triplets(df: pd.DataFrame, min_uc: int = 0, min_ic: int = 10):
    # df, min_uc, min_ic = data, 10, 10

    # Only keep the triplets for items which were played on by at least min_sc users.
    if min_ic > 0:
        itemcount = get_count(df, 'item')
        df = df[
            df['item'].isin(itemcount.loc[itemcount['size'] >= min_ic].item.to_list())
        ]

    # Only keep the triplets for users who clicked on at least min_uc items
    # After doing this, some of the items will have less than min_uc users, but should only be a small proportion
    if min_uc > 0:
        usercount = get_count(df, 'user')
        df = df[
            df['user'].isin(usercount.loc[usercount['size'] >= min_uc].user.to_list())
        ]

    # Update both usercount and itemcount after filtering
    usercount, itemcount = get_count(df, 'user'), get_count(df, 'item')
    return df, usercount, itemcount


def load_and_parse_msd_dataset():

    if not MSD_CSV_TRIPLETS_PATH.exists():
        helper = MSDDataHelper()
        columns = ['user', 'item', 'rating']

        rows = helper.get_data()
        data = pd.DataFrame.from_records(rows, columns=columns)
        data.rating = data.rating.astype('int32')
        data.to_csv(f'data/msd-triplets_artists.csv', sep=';', header=True, index=False)

    else:
        data = pd.read_csv(
            MSD_CSV_TRIPLETS_PATH,
            sep=';',
            dtype={'user': str, 'item': str, 'rating': 'int32'}
        )

    raw_data, user_activity, item_popularity = filter_triplets(data, min_uc=0, min_ic=128)

    unique_uid = user_activity.user
    idx_perm = np.random.permutation(unique_uid.size)
    unique_uid = unique_uid[idx_perm]

    n_users = unique_uid.size
    n_heldout_users = 10000

    true_users = unique_uid[:(n_users - n_heldout_users * 2)]
    vd_users = unique_uid[(n_users - n_heldout_users * 2): (n_users - n_heldout_users)]
    test_users = unique_uid[(n_users - n_heldout_users):]
    
    train_plays = raw_data.loc[raw_data.user.isin(true_users)]
    unique_tid = train_plays.item.unique()

    track2id = dict((sid, i) for (i, sid) in enumerate(unique_tid))
    user2id = dict((uid, i) for (i, uid) in enumerate(unique_uid))

    def split_train_test_proportion(
        plays: pd.DataFrame,
        test_prop: float = 0.25
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Randomly split plays data into train and test parts
        Train part of triplets is using to compute representation of user and top items to recommend
        Test part of triplets is using to evaluate recommendations
        """
        # plays, test_prop = vad_plays, 0.25
        data_grouped_by_user = plays.groupby('user')
        true_list, test_list = list(), list()

        for i, (_, group) in enumerate(data_grouped_by_user):
            # break

            # check if particular user has sufficient context to train/test on
            user_played_tracks = len(group)
            if user_played_tracks >= 8:
                test_part = group.sample(frac=test_prop, replace=False)
                true_part = group.loc[~group.item.isin(test_part.item)]
                assert len(true_part) + len(test_part) == user_played_tracks

                test_list.append(test_part)
                true_list.append(true_part)
            else:
                true_list.append(group)

        data_true = pd.concat(true_list)
        data_test = pd.concat(test_list)
        assert len(data_true) + len(data_test) == len(plays)
        return data_true, data_test

    vad_plays = raw_data.loc[raw_data['user'].isin(vd_users)]
    vad_plays = vad_plays.loc[vad_plays['item'].isin(unique_tid)]
    vad_plays_true, vad_plays_test = split_train_test_proportion(vad_plays)

    test_plays = raw_data.loc[raw_data['user'].isin(test_users)]
    test_plays = test_plays.loc[raw_data['item'].isin(unique_tid)]
    test_plays_true, test_plays_test = split_train_test_proportion(test_plays)

    def numericalize(df: pd.DataFrame) -> pd.DataFrame:
        uid = df['user'].map(lambda x: user2id[x])
        tid = df['item'].map(lambda x: track2id[x])
        return pd.DataFrame(data={'uid': uid, 'tid': tid}, columns=['uid', 'tid'])

    train_data = numericalize(train_plays)
    vad_data_true = numericalize(vad_plays_true)
    vad_data_test = numericalize(vad_plays_test)
    test_data_true = numericalize(test_plays_true)
    test_data_test = numericalize(test_plays_test)

    N_ITEMS = len(unique_tid)
    assert N_ITEMS == raw_data.item.nunique()

    def to_csc_train_split(numeric_plays: pd.DataFrame) -> sp.csc_matrix:
        plays_n_users = numeric_plays.uid.max() + 1
        rows, cols = numeric_plays['uid'], numeric_plays['tid']
        return sp.csc_matrix(
            (np.ones_like(rows), (rows, cols)),
            shape=(plays_n_users, N_ITEMS),
            dtype=np.float64
        )

    def to_csc_valid_test_splits(
        numeric_plays_true: pd.DataFrame,
        numeric_plays_test: pd.DataFrame
    ) -> Tuple[sp.csc_matrix, sp.csc_matrix, List[int], List[int]]:
        # numeric_plays_true, numeric_plays_test = vad_data_true, vad_data_test

        start_idx = min(numeric_plays_true['uid'].min(), numeric_plays_test['uid'].min())
        end_idx = max(numeric_plays_true['uid'].max(), numeric_plays_test['uid'].max())

        rows_true, cols_true = numeric_plays_true['uid'] - start_idx, numeric_plays_true['tid']
        rows_test, cols_test = numeric_plays_test['uid'] - start_idx, numeric_plays_test['tid']

        data_true = sp.csc_matrix(
            (np.ones_like(rows_true), (rows_true, cols_true)),
            shape=(end_idx - start_idx + 1, N_ITEMS),
            dtype=np.float64
        )

        data_test = sp.csc_matrix(
            (np.ones_like(rows_test), (rows_test, cols_test)),
            shape=(end_idx - start_idx + 1, N_ITEMS),
            dtype=np.float64
        )

        true_uids = np.unique(numeric_plays_true['uid']).tolist()
        test_uids = np.unique(numeric_plays_test['uid']).tolist()
        return data_true, data_test, true_uids, test_uids

    train_csc_data = to_csc_train_split(train_data)
    valid_csc_data_true, valid_csc_data_test, valid_uids_true, valid_uids_test = \
        to_csc_valid_test_splits(vad_data_true, vad_data_test)
    test_csc_data_true, test_csc_data_test, test_uids_true, test_uids_test = \
        to_csc_valid_test_splits(test_data_true, test_data_test)

    def test_splits(
        csc_splits: List[sp.csc_matrix],
        uids_splits: List[List[int]],
        orig_data: pd.DataFrame
    ) -> None:
        """
            Thou should write tests, bEAtch!
            Everytime you write a test you say "Thank You" to a future you
            """

        id2user = {idd: u for u, idd in user2id.items()}
        id2item = {idd: t for t, idd in track2id.items()}

        csc_split_true, csc_split_test = csc_splits
        uids_true, uids_test = uids_splits

        r, c = csc_split_test.nonzero()
        sampled_useridx = np.random.choice(np.unique(r), 25, replace=False)
        for useridx in sampled_useridx:
            orig_user_id = id2user[uids_true[useridx]]
            items = c[np.where(r == useridx)]
            orig_item_ids = [id2item[itemidx] for itemidx in items]
            known_item_ids = orig_data.loc[orig_data.user == orig_user_id].item.to_list()
            assert not set(orig_item_ids).difference(known_item_ids)

    test_splits(
        [valid_csc_data_true, valid_csc_data_test],
        [valid_uids_true, valid_uids_test],
        raw_data
    )
    test_splits(
        [test_csc_data_true, test_csc_data_test],
        [test_uids_true, test_uids_test],
        raw_data
    )

    # save splits as sparse matrices
    save_npz('train-data.npz', train_csc_data)
    save_npz('valid-data-true.npz', valid_csc_data_true)
    save_npz('valid-data-test.npz', valid_csc_data_test)
    save_npz('test-data-true.npz', test_csc_data_true)
    save_npz('test-data-test.npz', test_csc_data_test)

    # save mappings
    with MSD_SPLITS_DATA_PATH.joinpath('user2id.json').open(mode='w') as f:
        json.dump(user2id, f)
    with MSD_SPLITS_DATA_PATH.joinpath('item2id.json').open(mode='w') as f:
        json.dump(track2id, f)

    return \
        train_csc_data, valid_csc_data_true, valid_csc_data_test, \
        test_csc_data_true, test_csc_data_test
