import os, argparse, shutil, cv2, pickle, time, logging, gc, json
from utils.file import get_dirs
from unet.trainers import train_once, train_loop


dirs = get_dirs()
LOG_PATH     = dirs["files"][0]["LOG_PATH"]
RESULT_PATH  = dirs["files"][0]["RESULT_PATH"]
MODEL_PATH   = dirs["files"][0]["MODEL_PATH"]

TRAIN_PATH   = dirs["files"][0]["TRAIN_PATH"]
TEST_PATH    = dirs["files"][0]["TEST_PATH"]
VAL_PATH     = dirs["files"][0]["VAL_PATH"]

TMP_TRAIN    = dirs["files"][0]["TMP_TRAIN"]
TMP_TEST     = dirs["files"][0]["TMP_TEST"]
TMP_VAL      = dirs["files"][0]["TMP_VAL"]
TMP_RESULT   = dirs["files"][0]["TMP_RESULT"]


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--train_at_once", "-tao", help="--train_at_once=False allows you to \
                          save models and predictions at every epoch. True is default. If True, \
                          then your results and checkpoints are saved at the end.", type = bool, default = True)

    parser.add_argument("--save_name", "-sn", help="Your experiment's name. It is related to\
                          saved checkpoints, folders etc.", type = str, default = "hello_world")

    parser.add_argument("--initial_model_path", "-initm", help="Previous checkpoints to load. If not, default is None.", default = None)
    parser.add_argument("--model_name", "-m", help="Which unet to use.", type=str, default = "vanilla")
    parser.add_argument("--epochs", "-e", help="Number of epochs", type=int)
    parser.add_argument("--train_batch", "-tb", help="Training batch size.", default = 3, type=int)
    parser.add_argument("--val_batch", "-vb", help="Validation batch size.", default = 3,type=int)
    parser.add_argument("--already_padded", "-ap", help="If your training/valiadion samples \
                          in your related folder have already been padded, no need to pad again.", default = False)
    args = parser.parse_args()


    train_sample_number = len(os.listdir(TRAIN_PATH + '/images'))
    test_sample_number  = len(os.listdir(TEST_PATH + '/images'))

    if args.train_at_once:

        train_once(save_name = args.save_name, initial_model_path = args.initial_model_path, \
                   epoch= args.epochs, train_batch = args.train_batch, test_batch = args.val_batch, \
                   model_name = args.model_name, already_padded = args.already_padded,\
                   num_train = train_sample_number, num_test= test_sample_number)

    elif not args.train_at_once:

        train_loop(save_name = args.save_name, initial_model_path = args.initial_model_path, \
                   epoch= args.epochs, train_batch = args.train_batch, test_batch = args.val_batch, \
                   model_name = args.model_name, already_padded = args.already_padded,\
                   num_train = train_sample_number, num_test= test_sample_number)



