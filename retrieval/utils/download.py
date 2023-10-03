import argparse
import os
import tarfile
import urllib.request

from loguru import logger


def create_dir(_dir):
    if not os.path.isdir(_dir):
        os.mkdir(_dir)


def download_oxford_paris(data_dir):
    """
        Download [oxford5k, paris6k, roxford5k, rparis6k] test datasets

    source: https://github.com/filipradenovic/cnnimageretrieval-pytorch/blob/master/cirtorch/utils/download.py

        with minor modifications.
    """

    datasets = ['oxford5k', 'paris6k', 'roxford5k', 'rparis6k']

    # create data folder, if it does not exist
    create_dir(data_dir)

    # create datasets folder, if it does not exist
    datasets_dir = os.path.join(data_dir, 'test')
    create_dir(datasets_dir)

    # run
    for di in range(len(datasets)):
        dataset = datasets[di]

        if dataset == 'oxford5k':
            src_dir = 'https://www.robots.ox.ac.uk/~vgg/data/oxbuildings'
            dl_files = ['oxbuild_images-v1.tgz']
        elif dataset == 'paris6k':
            src_dir = 'https://www.robots.ox.ac.uk/~vgg/data/parisbuildings'
            dl_files = ['paris_1-v1.tgz', 'paris_2-v1.tgz']
        elif dataset == 'roxford5k':
            src_dir = 'https://www.robots.ox.ac.uk/~vgg/data/oxbuildings'
            dl_files = ['oxbuild_images-v1.tgz']
        elif dataset == 'rparis6k':
            src_dir = 'https://www.robots.ox.ac.uk/~vgg/data/parisbuildings'
            dl_files = ['paris_1-v1.tgz', 'paris_2-v1.tgz']
        else:
            raise ValueError('Unknown dataset: {}!'.format(dataset))

        dst_dir = os.path.join(datasets_dir, dataset, 'jpg')
        if not os.path.isdir(dst_dir):

            # for oxford and paris download images
            if dataset == 'oxford5k' or dataset == 'paris6k':
                print('>> Dataset {} directory does not exist. Creating: {}'.format(
                    dataset, dst_dir))
                os.makedirs(dst_dir)
                for dli in range(len(dl_files)):
                    dl_file = dl_files[dli]
                    src_file = os.path.join(src_dir, dl_file)
                    dst_file = os.path.join(dst_dir, dl_file)

                    # download
                    logger.info(
                        f'downloading dataset {dataset} archive {dl_file}')
                    os.system(f'wget {src_file} -O {dst_file}')

                    # extract
                    logger.info(
                        f'extracting dataset {dataset} archive {dl_file}')

                    # create tmp folder
                    dst_dir_tmp = os.path.join(dst_dir, 'tmp')
                    os.system('mkdir {}'.format(dst_dir_tmp))

                    # extract in tmp folder
                    os.system('tar -zxf {} -C {}'.format(dst_file, dst_dir_tmp))

                    # remove all (possible) subfolders by moving only files in dst_dir
                    os.system(
                        'find {} -type f -exec mv -i {{}} {} \\;'.format(dst_dir_tmp, dst_dir))

                    # remove tmp folder
                    os.system('rm -rf {}'.format(dst_dir_tmp))
                    logger.info(
                        f'Extraction done, dataset {dataset} archive {dl_file}')
                    os.system('rm {}'.format(dst_file))

            # symbol links
            elif dataset == 'roxford5k' or dataset == 'rparis6k':
                logger.info(
                    f'dataset {dataset} directory does not exist. Creating: {dst_dir}')

                dataset_old = dataset[1:]
                dst_dir_old = os.path.join(datasets_dir, dataset_old, 'jpg')
                os.mkdir(os.path.join(datasets_dir, dataset))

                os.system('ln -s {} {}'.format(dst_dir_old, dst_dir))
                logger.info(
                    'created symbolic link from {dataset_old} jpg to {dataset} jpg')

        gnd_src_dir = os.path.join(
            'https://cmp.felk.cvut.cz/cnnimageretrieval/data', 'test', dataset)
        gnd_dst_dir = os.path.join(datasets_dir, dataset)
        gnd_dl_file = 'gnd_{}.pkl'.format(dataset)
        gnd_src_file = os.path.join(gnd_src_dir, gnd_dl_file)
        gnd_dst_file = os.path.join(gnd_dst_dir, gnd_dl_file)

        if not os.path.exists(gnd_dst_file):
            logger.info('downloading dataset {dataset} ground truth file')
            os.system('wget {} -O {}'.format(gnd_src_file, gnd_dst_file))


def download_sfm120(data_dir):
    """
        Download [retrieval-SfM-120k retrieval-SfM-30k] train datasets

    source: https://github.com/filipradenovic/cnnimageretrieval-pytorch/blob/master/cirtorch/utils/download.py

        with minor modifications.      
    """
    datasets = ['retrieval-SfM-120k', 'retrieval-SfM-30k']

    # create data folder, if it does not exist
    create_dir(data_dir)

    # create datasets folder, if it does not exist
    datasets_dir = os.path.join(data_dir, 'train')
    create_dir(datasets_dir)

    # download folder train
    src_dir = os.path.join(
        'https://cmp.felk.cvut.cz/cnnimageretrieval/data', 'train', 'ims')
    dst_dir = os.path.join(datasets_dir, 'retrieval-SfM-120k', 'ims')

    dl_file = 'ims.tar.gz'

    if not os.path.isdir(dst_dir):
        src_file = os.path.join(src_dir, dl_file)
        dst_file = os.path.join(dst_dir, dl_file)

        # create
        os.makedirs(dst_dir)

        # download
        logger.info(f'downloading {dst_dir}')
        os.system('wget {} -O {}'.format(src_file, dst_file))

        logger.info(f'extracting {dst_file}')
        os.system('tar -zxf {} -C {}'.format(dst_file, dst_dir))

        logger.info(f'extraction done, deleting {dst_file}')
        os.system('rm {}'.format(dst_file))

    # symlink for train/retrieval-SfM-30k/
    dst_dir_old = os.path.join(datasets_dir, 'retrieval-SfM-120k', 'ims')
    dst_dir = os.path.join(datasets_dir, 'retrieval-SfM-30k', 'ims')

    if not os.path.isdir(dst_dir):
        os.makedirs(os.path.join(datasets_dir, 'retrieval-SfM-30k'))
        os.system('ln -s {} {}'.format(dst_dir_old, dst_dir))
        logger.info('created symbolic link to retrieval-SfM-30k/ims')

    # download db files
    src_dir = os.path.join(
        'https://cmp.felk.cvut.cz/cnnimageretrieval/data', 'train', 'dbs')
    for dataset in datasets:
        dst_dir = os.path.join(datasets_dir, dataset)
        if dataset == 'retrieval-SfM-120k':
            dl_files = ['{}.pkl'.format(
                dataset), '{}-whiten.pkl'.format(dataset)]
        elif dataset == 'retrieval-SfM-30k':
            dl_files = ['{}-whiten.pkl'.format(dataset)]

        if not os.path.isdir(dst_dir):
            logger.info(f'Creating: {dst_dir}')
            os.mkdir(dst_dir)

        for i in range(len(dl_files)):
            src_file = os.path.join(src_dir, dl_files[i])
            dst_file = os.path.join(dst_dir, dl_files[i])

            if not os.path.isfile(dst_file):
                logger.info(f'downloading db file {dl_files[i]}')
                os.system('wget {} -O {}'.format(src_file, dst_file))


def download_revisited1m(data_dir):
    """
    DOWNLOAD_DISTRACTORS Checks, and, if required, downloads the distractor dataset.
    download_distractors(DATA_ROOT) checks if the distractor dataset exist.
    If not it downloads it in the folder:
        DATA_ROOT/datasets/revisitop1m/   : folder with 1M distractor images
    """

    # create data folder if it does not exist
    create_dir(data_dir)

    # create datasets folder if it does not exist
    datasets_dir = os.path.join(data_dir, 'datasets')
    create_dir(datasets_dir)

    dataset = 'revisitop1m'
    datasets_dir = os.path.join(datasets_dir, dataset)
    create_dir(datasets_dir)

    nfiles = 100
    src_dir = 'http://ptak.felk.cvut.cz/revisitop/revisitop1m/jpg'
    dl_files = 'revisitop1m.{}.tar.gz'

    dst_dir = os.path.join(datasets_dir, 'jpg')
    dst_dir_tmp = os.path.join(datasets_dir, 'jpg_tmp')

    if not os.path.isdir(dst_dir):
        logger.info(f'creating dataset {dataset} in : {dst_dir}')

        create_dir(dst_dir_tmp)

        for dfi in range(nfiles):
            dl_file = dl_files.format(dfi+1)
            src_file = os.path.join(src_dir, dl_file)
            dst_file = os.path.join(dst_dir_tmp, dl_file)
            dst_file_tmp = os.path.join(dst_dir_tmp, dl_file + '.tmp')
            if os.path.exists(dst_file):
                logger.info(
                    f'[{dfi+1}/{nfiles}] skip dataset {dataset} archive {dl_file} already exists')
            else:
                while 1:
                    try:
                        logger.info(
                            f'[{dfi+1}/{nfiles}] downloading dataset {dataset} archive {dl_file}')
                        urllib.request.urlretrieve(src_file, dst_file_tmp)
                        os.rename(dst_file_tmp, dst_file)
                        break
                    except Exception as e:
                        logger.warning(
                            f'download failed. try this one again. Error: {e}')
                        logger.warning('download failed. try this one again')

        for dfi in range(nfiles):
            dl_file = dl_files.format(dfi+1)

            dst_file = os.path.join(dst_dir_tmp, dl_file)
            logger.info(
                f'[{dfi+1}/{nfiles}] extracting dataset {dataset} archive {dl_file}')

            tar = tarfile.open(dst_file)
            tar.extractall(path=dst_dir_tmp)
            tar.close()

            logger.info(
                f'[{dfi+1}/{nfiles}] extracted, deleting dataset {dataset} archive {dl_file}')
            os.remove(dst_file)

        # rename tmp folder
        os.rename(dst_dir_tmp, dst_dir)

        # download image list
        gnd_src_dir = 'http://ptak.felk.cvut.cz/revisitop/revisitop1m/'
        gnd_dst_dir = os.path.join(data_dir, 'datasets', dataset)
        gnd_dl_file = '{}.txt'.format(dataset)
        gnd_src_file = os.path.join(gnd_src_dir, gnd_dl_file)
        gnd_dst_file = os.path.join(gnd_dst_dir, gnd_dl_file)
        if not os.path.exists(gnd_dst_file):
            logger.info(f'downloading dataset {dataset} image list file')
            urllib.request.urlretrieve(gnd_src_file, gnd_dst_file)


if __name__ == '__main__':

    # ArgumentParser
    parser = argparse.ArgumentParser(description='VISLOC:: Dataset download')
    parser.add_argument('--data',       metavar='EXPORT_DIR',
                        default='.',    help='dataset folder')
    parser.add_argument("--dataset",   type=str,   default='all',
                        help='(oxford_paris, sfm120, revisited1m, all]')

    args = parser.parse_args()

    if args.dataset == 'oxford_paris':
        download_oxford_paris(args.data)
    elif args.dataset == 'sfm120':
        download_sfm120(args.data)
    elif args.dataset == 'revisited1m':
        download_revisited1m(args.data)
    elif args.dataset == 'all':
        download_oxford_paris(args.data)
        download_sfm120(args.data)
        download_revisited1m(args.data)
