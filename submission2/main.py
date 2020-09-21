from sklearn.metrics import precision_recall_curve
import numpy as np
import argparse
from dataloader import get_audio_loader
from solver import Solver

PATH = '/home/toby/MSc_AI/IndividualProject/music_gen_evaluation_framework/data/datasets/mtg-jamendo-dataset'
TRAIN_PATH = f'{PATH}/mood-theme/mood_clustering/match_train_labels.csv'
VAL_PATH = f'{PATH}/mood-theme/mood_clustering/match_validation_labels.csv'
TEST_PATH = f'{PATH}/mood-theme/mood_clustering/match_test_labels.csv'
DATA_PATH = f'{PATH}/mood-theme/melspecs'
LOG_PATH = '/home/toby/MSc_AI/IndividualProject/music_gen_evaluation_framework/audio_tagging_implementations/mediaeval-2019-moodtheme-detection/submission2/output'
PRETRAINED_MODEL_PATH = f'{LOG_PATH}/pretrained_baseline.pth'
LABELS_TXT = f'{PATH}/mood-theme/mood_clustering/emotional_valence.txt'

def get_labels_to_idx(labels_txt):
    labels_to_idx = {}
    tag_list = []
    with open(labels_txt) as f:
        lines = f.readlines()

    for i,l in enumerate(lines):
        tag_list.append(l.strip())
        labels_to_idx[l.strip()] = i

    return labels_to_idx, tag_list

def train(exp_name,mode=None,num_epochs=120):
    config = CONFIG
    
    labels_to_idx, tag_list = get_labels_to_idx(LABELS_TXT)    
    train_loader1 = get_audio_loader(DATA_PATH_TRAIN, TRAIN_PATH, labels_to_idx, batch_size=config['batch_size'])
    train_loader2 = get_audio_loader(DATA_PATH_TRAIN, TRAIN_PATH, labels_to_idx, batch_size=config['batch_size'])
    print(VAL_PATH)
    val_loader = get_audio_loader(DATA_PATH_TEST, VAL_PATH, labels_to_idx, batch_size=config['batch_size'], shuffle=False, drop_last=False)
    
    solver = Solver(train_loader1,train_loader2, val_loader, tag_list, config,LABELS_TXT, num_epochs=num_epochs, mode=mode)
    solver.train(exp_name)

def predict(model_fn,mode=None):
    config = CONFIG
    labels_to_idx, tag_list = get_labels_to_idx(LABELS_TXT)

    test_loader = get_audio_loader(DATA_PATH_TEST, TEST_PATH, labels_to_idx, batch_size=config['batch_size'], shuffle=False, drop_last=False)

    solver = Solver(test_loader,None, None, tag_list, config,LABELS_TXT, num_epochs=120, mode=mode)
    predictions = solver.test(model_fn)

    np.save(f"{CONFIG['log_dir']}/predictions.npy", predictions)

def parseArguments():
    # Set paths

    # Create argument parser
    parser = argparse.ArgumentParser()

    # Positional mandatory arguments
    parser.add_argument("test_or_train", help="Test or train or fine_tune mode", choices=['test','train'])
    parser.add_argument("exp_name", help="name of saved model")
    # Optional Arguments
    parser.add_argument("-train", "--train_datapath", help="Path to train csv",default=TRAIN_PATH)
    parser.add_argument("-test","--test_datapath", help="Path to test csv",default=TEST_PATH)
    parser.add_argument("-val", "--val_datapath", help="Path to val csv", default=VAL_PATH)
    parser.add_argument("-log","--log_datapath", help="Where to store logs",default=LOG_PATH)
    parser.add_argument("-labels","--labels_path", help="Path to list of labels",default=LABELS_TXT)
    parser.add_argument("-data","--datapath", help="Path to training melspecs data",default=DATA_PATH)
    parser.add_argument("-model","--model_path", help="Path to model",default=PRETRAINED_MODEL_PATH)
    parser.add_argument("-test_val_data","--test_val_data", help="Path to test melspecs data",default=DATA_PATH)

    # Parse arguments
    args = parser.parse_args()

    return args

if __name__=="__main__":
    args = parseArguments()

    mode = args.__dict__['test_or_train']


    DATA_PATH_TRAIN = args.datapath  # Path to melspecs
    LABELS_TXT = args.labels_path  # list of labels
    TRAIN_PATH = args.train_datapath  # audio paths and tags
    TEST_PATH = args.test_datapath
    VAL_PATH = args.val_datapath
    DATA_PATH_TEST = args.test_val_data
    LOG_PATH = args.log_datapath
    
    CONFIG = {
        'log_dir': LOG_PATH,
        'batch_size': 8
    }
    if mode == 'train':
        train(args.exp_name)

    elif mode == 'test':

        predict(args.model_path)
