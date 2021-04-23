import os, argparse, shutil, cv2, pickle, time, logging, gc, json
from utils.file import get_dirs
from unet.trainers import train_once, train_loop, train_kfold_stare


dirs = get_dirs()
TRAIN_PATH_IMG        = dirs["files"][0]["TRAIN_PATH"] + "/images"
TRAIN_PATH_MASK       = dirs["files"][0]["TRAIN_PATH"] + "/labels"
KFOLD_TEMP_TRAIN      = dirs["files"][0]["KFOLD_TEMP_TRAIN"]
KFOLD_TEMP_TEST       = dirs["files"][0]["KFOLD_TEMP_TEST"]

LOG_PATH_K   = dirs["files"][0]["LOG_PATH_KFOLD"]
CKPTS_PATH = dirs["files"][0]["CKPTS_PATH_KFOLD"]
RESULTS_PATH = dirs["files"][0]["RESULTS_PATH_KFOLD"]



if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--save_name", "-sn", help="Your experiment's name. It is related to\
                          saved checkpoints, folders etc.", type = str, default = "hello_world")

    parser.add_argument("--initial_model_path", "-initm", help="Previous checkpoints to load. If not, default is None.", default = None)

    parser.add_argument("--epochs", "-e", help="Number of epochs", type=int)
    parser.add_argument("--train_batch", "-tb", help="Training batch size.", default = 3, type=int)
    parser.add_argument("--val_batch", "-vb", help="Validation batch size.", default = 3,type=int)
    parser.add_argument("--n_fold", "-nf", help="Your number of folds.", default = 4, type=int)
    parser.add_argument("--start_fold", "-sf", help="Where to start your fold.", default = 0, type=int)
    parser.add_argument("--show_samples", "-ss", help="Show predicted masks in validation set while training.", default = False, type=bool)
    args = parser.parse_args()


    train_sample_number = len(os.listdir(TRAIN_PATH_MASK)) - (len(os.listdir(TRAIN_PATH_MASK))//20*args.n_fold)


    train_kfold_stare(epoch = args.epochs, start = args.start_fold, \
                      train_batch_size = args.train_batch, \
                      test_batch_size = args.val_batch,\
                      train_sample_number = train_sample_number,\
                      test_sample_number = args.n_fold, \
                      initial_model_path = args.initial_model_path,\
                      k = args.n_fold, \
                      show_samples = args.show_samples)



