import argparse
import csv
import os
import pickle

import numpy as np
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
        description='Prepare Image Google Landmark 2018')

    parser.add_argument('--data', metavar='EXPORT_DIR',
                        help='path to train / val GL18 dataste')

    args = parser.parse_args()

    for arg in vars(args):
        print(' {}\t  {}'.format(arg, getattr(args, arg)))

    print('\n ')

    return parser


def main(args):

    print(" Prepare Google Landmark 2018 : ", args.data)

    db_csv = os.path.join(args.data, 'train.csv')

    train_split_csv = os.path.join(args.data, 'boxes_split1.csv')
    val_split_csv = os.path.join(args.data, 'boxes_split2.csv')

    # Read database
    db_file = open(db_csv, 'r')
    db_file = csv.reader(db_file)

    key_landmark_list = [[line[0]] + [line[-1]] for line in db_file]

    all_images_list = [line[0] for line in key_landmark_list]
    key_landmark_dict = {line[0]: line[-1] for line in key_landmark_list}

    # Train split
    train_csv = open(train_split_csv, 'r')
    train_file = csv.reader(train_csv)
    train_split = [line for line in train_file][1:]

    # Val split
    val_csv = open(val_split_csv, 'r')
    val_file = csv.reader(val_csv)
    val_split = [line for line in val_file][1:]

    train_split_dict = {}
    for _id, box in train_split:
        train_split_dict[_id] = box

    val_split_dict = {}
    for _id, box in val_split:
        val_split_dict[_id] = box

    val_list_all = list(val_split_dict.keys())
    train_list_all = list(set(all_images_list) - set(val_list_all))

    train_list = []
    val_list = []

    landmark_ids = {}
    landmark_ids['train'] = []
    landmark_ids['val'] = []

    train_and_val_list = {}
    train_and_val_list['train'] = []
    train_and_val_list['val'] = []

    # Search for exsisting training images
    print(" Finding all train images that succesfully downloaded")
    for cid in tqdm(train_list_all):
        if os.path.exists(os.path.join(args.data, 'train', cid + '.jpg')):
            train_list.append(cid)
            landmark_ids['train'].append(key_landmark_dict[cid])

    print("Training images found :", len(train_list))

    train_and_val_list['train'] = train_list

    # Search for exsisting validation  images
    print("Finding all val images that succesfully downloaded")
    for cid in tqdm(val_list_all):
        if os.path.exists(os.path.join(args.data, 'train', cid + '.jpg')):
            val_list.append(cid)
            landmark_ids['val'].append(key_landmark_dict[cid])

    print("Validation images found :", len(val_list))

    train_and_val_list['val'] = val_list

    # extra
    train_list = [cid for cid in train_list_all if os.path.exists(
        os.path.join(args.data, 'train', cid + '.jpg'))]
    val_list = [cid for cid in val_list_all if os.path.exists(
        os.path.join(args.data, 'train', cid + '.jpg'))]

    key_landmark_list = key_landmark_list[1:]  # Chop off header

    _id_to_landmark = {}
    _id_to_landmark['train'] = {}
    _id_to_landmark['val'] = {}

    landmark_to_id = {}
    landmark_to_id['train'] = {}
    landmark_to_id['val'] = {}

    db_dict = {}
    db_dict['train'] = {}
    db_dict['val'] = {}

    boxes = {}
    boxes['train'] = train_split_dict
    boxes['val'] = val_split_dict

    db_dict['train']['cids'] = []
    db_dict['train']['qidxs'] = []
    db_dict['train']['pidxs'] = []
    db_dict['train']['cluster'] = []
    db_dict['train']['bbxs'] = []

    db_dict['val']['cids'] = []
    db_dict['val']['qidxs'] = []
    db_dict['val']['pidxs'] = []
    db_dict['val']['cluster'] = []
    db_dict['val']['bbxs'] = []

    _id_dict = {}
    _id_dict['train'] = {}
    _id_dict['val'] = {}

    i = 0

    unique_landmarks = {}
    unique_landmarks['train'] = list(set(list(landmark_ids['train'])))
    unique_landmarks['val'] = list(set(list(landmark_ids['val'])))

    landmark_to_qids = {}
    landmark_to_qids['train'] = {}
    landmark_to_qids['val'] = {}

    # Finding idxs that corresponds to each landmark
    print('finding idxs that corresponds to each landmark...')

    for mode in ['train', 'val']:
        for landmark in tqdm(unique_landmarks[mode]):
            landmark_to_qids[mode][landmark] = np.where(
                np.array(landmark_ids[mode]) == landmark)[0].tolist()

    for mode in ['train', 'val']:

        image_list = train_list if mode == 'train' else val_list

        boxes[mode]

        for i, image in enumerate(tqdm(image_list)):

            db_dict[mode]['cids'].append(image)

            landmark = key_landmark_dict[image]

            db_dict[mode]['cluster'].append(landmark)

            pidxs_potential = landmark_to_qids[mode][landmark]

            try:
                pidxs_potential.remove(image)
            except ValueError:
                pass

            try:
                pidxs_potential.remove(i)
            except ValueError:
                pass

            if len(pidxs_potential) == 0:
                continue

            # not a list
            pidxs = np.random.choice(
                pidxs_potential, min(len(pidxs_potential), 1))

            try:
                bbx = list(map(float, train_split_dict[image].split()))
                db_dict[mode]['bbxs'].append(bbx)

            except ValueError:
                db_dict[mode]['bbxs'].append(None)

            db_dict[mode]['qidxs'].append(i)
            db_dict[mode]['pidxs'].append(pidxs[0])

    print('Save pkl file')
    pkl_path = os.path.join(args.data,  'gl18.pkl')
    pickle.dump(db_dict, open(pkl_path, 'wb'))


if __name__ == '__main__':

    parser = make_parser()

    main(parser.parse_args())
