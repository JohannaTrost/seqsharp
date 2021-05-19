import json, os, errno, math
from datetime import datetime

import numpy as np


def write_config_file(nb_protein_families,
                      min_seqs_per_align,
                      max_seqs_per_align,
                      seq_len, batch_size,
                      epochs, lr, optimizer,
                      nb_folds, model_path):
    PARAMS = {"data":
        {
            "nb_protein_families": nb_protein_families,
            "min_seqs_per_align": min_seqs_per_align,
            "max_seqs_per_align": max_seqs_per_align,
            "seq_len": seq_len,
        },
        "hyperparams":
            {
                "batch_size": batch_size,
                "epochs": epochs,
                "lr": lr,
                "optimizer": optimizer,
                "nb_folds": nb_folds
            },
        "results_path": model_path,
        "comments" : ""
    }

    timestamp = datetime.now().strftime("%d-%b-%Y-%H:%M:%S.%f")
    with open("configs/config-" + timestamp + ".json", "w") as outfile:
        json.dump(PARAMS, outfile)


def read_config_file(path):

    if os.path.exists(path):

        if os.path.isdir(path):
            path = path + '/' + os.listdir(path)[-1]

        with open(path) as json_file:
            params = json.load(json_file)
    else:
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT),
                                path)

    return params


def split_lst(lst, nb_parts):
    n = math.ceil(len(lst) / nb_parts)
    for i in range(0, len(lst), n):
        yield lst[i: n + i]


def flatten_lst(lst):
    flattened_lst = []
    for _lst in lst:
        flattened_lst += _lst
    return flattened_lst

