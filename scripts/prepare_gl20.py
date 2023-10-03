import argparse
import glob as glob
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

np.random.seed(123)


def _id_to_qid(_id_list, _id_dict_):
    qid_list = []
    for _id in tqdm(_id_list):
        qid_list.append(_id_dict_[_id])

    return qid_list


def make_parser():
    # ArgumentParser
    parser = argparse.ArgumentParser(
        description='Prepare Google Landmark 2020')

    parser.add_argument('--data', metavar='EXPORT_DIR',
                        help='path to train / val GL20 dataste')

    args = parser.parse_args()

    for arg in vars(args):
        print(' {}\t  {}'.format(arg, getattr(args, arg)))
    print('\n ')
    return parser


def main(args):

    print(" Prepare Google Landmark 2020 : ", args.data)

    # train
    train_csv = os.path.join(args.data,   'train.csv')
    train_db = pd.read_csv(train_csv)

    # find all train images
    train_imgs_path = Path(args.data, 'train').glob('**/*.jpg')
    imgs_db = pd.DataFrame(train_imgs_path, columns=['path'])

    # TODO: exclude args.data
    imgs_db['path'] = imgs_db['path'].apply(lambda x: str(x.absolute()))
    imgs_db['id'] = imgs_db['path'].apply(
        lambda x: x.split('/')[-1].replace('.jpg', ''))

    # merge by id
    train_db = train_db.merge(imgs_db, on='id')
    print(train_db)

    db_dict = {}
    db_dict['train'] = {}

    db_dict['train']['cids'] = []
    db_dict['train']['qidxs'] = []
    db_dict['train']['pidxs'] = []
    db_dict['train']['cluster'] = []
    db_dict['train']['bbxs'] = []

    # train_db_pkl_path = os.path.join(args.data, "train.pkl")
    # train_db.to_pickle(train_db_pkl_path)
    # print(f"save train {train_db_pkl_path}")

    # # class frequency
    # class_freq = train_db.landmark_id.value_counts()
    # np.savetxt(os.path.join(args.data, 'class_freq.txt'), class_freq.values, fmt='%d')

    # # topk
    # class_topk=1000
    # use_landmark_ids = class_freq.index[:class_topk]

    # keep_index = []
    # limit_samples_per_class = 500

    # # balance and filter
    # if limit_samples_per_class > 0:  #

    #     #
    #     for id_ in class_freq[class_freq > limit_samples_per_class].index:
    #         idx = train_db.query('landmark_id == @id_').sample(n=limit_samples_per_class, random_state=12345).index
    #         keep_index.extend(idx)

    #     large_class_index = class_freq[class_freq > limit_samples_per_class].index
    #     use_landmark_ids  = pd.Index(set(use_landmark_ids) - set(large_class_index))

    #     keep_index = pd.Index(keep_index)
    #     keep_index = keep_index.append(train_db[train_db['landmark_id'].isin(use_landmark_ids)].index)

    # else:
    #     #
    #     keep_index = train_db['landmark_id'].isin(use_landmark_ids)

    # # filtred db
    # train_filtered                  = train_db.loc[keep_index]
    # train_filtered['landmark_id']   = train_filtered['landmark_id'].astype('category')
    # train_filtered['class_id']      = train_filtered['landmark_id'].cat.codes.astype('int64')

    # #
    # class_freq = train_filtered.landmark_id.value_counts()
    # np.savetxt(os.path.join(args.data, 'filtred_class_freq.txt'), class_freq.values, fmt='%d')

    # # save filtred
    # train_filtered_pkl_path = os.path.join(args.data, "train_filtred.pkl")
    # train_filtered.to_pickle(train_filtered_pkl_path)
    # print(f"save filtred {train_filtered_pkl_path}")

    # train clean
    train_clean_csv = os.path.join(args.data,   'train_clean.csv')
    train_clean_db = pd.read_csv(train_clean_csv)

    print(train_clean_db)

    clean_train_ids = np.concatenate(train_clean_db['images'].str.split(' '))

    new_train_db = train_db[train_db['id'].isin(clean_train_ids)]

    print(new_train_db)

    assert len(train_clean_db) == new_train_db['landmark_id'].nunique()

    # landmark to images  dict
    landmark_to_images_dict = {}
    landmark_to_images_list = zip(
        train_clean_db['landmark_id'].to_numpy(), train_clean_db['images'])

    #
    for item_cls, item_imgs in landmark_to_images_list:
        landmark_to_images_dict[item_cls] = item_imgs.split(' ')

    print(f"we have {len(landmark_to_images_dict)} landmark")

    #
    db_dict = {}
    db_dict['train'] = {}
    db_dict['train']['cids'] = []
    db_dict['train']['qidxs'] = []
    db_dict['train']['pidxs'] = []
    db_dict['train']['cluster'] = []
    db_dict['train']['bbxs'] = []

    #
    data = new_train_db['id'].to_numpy()

    #
    id_to_name = {}

    for it, (q_id) in enumerate(tqdm(data, total=len(new_train_db))):
        id_to_name[it] = q_id

    name_to_id = {k: i for i, k in id_to_name.items()}

    #
    data = zip(new_train_db['id'].to_numpy(
    ), new_train_db['landmark_id'].to_numpy(), new_train_db['path'].to_numpy())

    for it, (q_id, cls, cid) in enumerate(tqdm(data, total=len(new_train_db))):

        #
        positive_keys_candidates = landmark_to_images_dict[cls]

        # filter keys
        positive_keys = []
        for k in positive_keys_candidates:
            if k in name_to_id.keys():
                positive_keys.append(k)

        # continue
        if len(positive_keys) < 1:
            print("no positives taken")
            continue

        # center key
        # m_key    = int( len(positive_keys) / 2.0)
        # pos_name = positive_keys[m_key]

        pos_ids = []
        for pos_name in positive_keys:
            pos_ids.append(name_to_id[pos_name])

        #
        rel_cid = os.path.relpath(cid, os.path.join(
            args.data, "train")).split(".")[0]

        #
        db_dict['train']['qidxs'].append(it)
        db_dict['train']['pidxs'].append(pos_ids)
        db_dict['train']['cluster'].append(cls)
        db_dict['train']['cids'].append(rel_cid)
        db_dict['train']['bbxs'].append(None)

    print(len(db_dict['train']['cids']))
    # save
    pkl_path = os.path.join(args.data,  'gl20.pkl')
    pickle.dump(db_dict, open(pkl_path, 'wb'))

    print("Done")


if __name__ == '__main__':

    #
    parser = make_parser()

    #
    main(parser.parse_args())
