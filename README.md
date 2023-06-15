# Seqsharp: Sequence Simulations Have A Real(ism) Problem

This is a tool that uses Convolutional Neural Networks (CNNs) to discriminate simulated and empirical multiple sequence alignments (MSAs) based on their site-wise compositions. The classifier's accuracy, measured as the Balanced Accuracy (BACC), that is the average of accuracies per class, serves as a metric to assess the realism of sequence evolution simulations.

Key components:

- `seqsharp/pretrained_models`: Here we provide the pre-trained seq#-models presented in our paper. They were trained on simulations using various evolutionary models for both DNA and protein sequences. In addition to the pretrained CNNs there are also pre-trained logistic regression models for all protein simulation settings. These were trained on average MSA compositions. 
- The modules included in the seqsharp-package enable training, testing, and evaluation of (pre-trained) seq#-models. They can be used directly or through the CLI. Below, you will find commands to reproduce the results in our paper.
- `seqsharp/compare_models.ipynb`: This Jupyter notebook offers a comprehensive analysis and comparison of the performance of our pre-trained models. It generates tables and figures that are included in the supplementary material of our paper.
- `protein_simulations`: This folder includes a CLI that allows to reproduce the simulated protein data collections of our study. 

To reproduce the results of the classifier using Gradient Boosted Trees please refer to [Julia's repo](https://github.com/tschuelia/SimulationStudy.git).

### Install 

A python version <= 3.6 is required for this package. 
As of now the maximum python version that allows to install all required packages is 3.10.

1. Clone the repository: `git clone https://github.com/JohannaTrost/seqsharp.git
2. Then navigate to the repository folder: `cd seqsharp` 
3. You can now install seq# using: `pip install . -r requirements.txt`
Note that for python v3.10 you might need to add `--use-pep517`

### Usage

To run seq# type:

`seqsharp [args]`

Type `seqsharp --help` to obtain the output below for possible arguments:

```
usage: seqsharp [-h] [--sim [SIM ...]] [--emp EMP] [-t] [--test] [--validate] [--attr] [--clr] [-m MODEL] [-c CFG] [-s SAVE] [--shuffle] [--ncpus NCPUS]

optional arguments:
  -h, --help            show this help message and exit
  --sim [SIM ...]       Specify the <path/to/> directory(s) containing simulated alignments (in fasta or phylip format)
  --emp EMP             Specify the <path/to/> directory containing empirical alignments (in fasta or phylip format)
  -t, --train           Data collections will be used to train the neural network (specified with --emp and --sim option). Requires --cfg or --model. If a pretrained model is
                        given training will be resumed.
  --test                Test network on given data collections. Requires --models and --sim
  --validate            K-fold cross validation with pretrained model.Requires --models, --emp and --sim
  --attr                Generates attribution maps using validation data. Requires --models (only if not --train) and --sim
  --clr                 Use cyclic learning rates (in this case lr given in config is ignored). Requires --train.
  -m MODEL, --model MODEL
                        <path/to> directory with pretrained model(s) (one per fold).
  -c CFG, --cfg CFG     <path/to> cfg file (.json) or directory containing: hyperparameters for training, data specific parameters and parameters for the network
                        architecture. Is not required when --model is given.
  -s SAVE, --save SAVE  <path/to> directory where trained models and result plots/tables will be saved.
  --shuffle             Shuffle the sites of alignments in the data collections.
  --ncpus NCPUS         Number of CPUs to be used.
```

### Reproduce pre-trained models 

To reproduce the training of the models presented in the paper use the following command:

`seqsharp -t --emp <path/to/emp/data> --sim <path/to/sim/data> -c <path/to/>seqsharp/seqsharp/pretrained_models/<evomodel>/cfg.json -s <path/to/results>
`

Replace `<evomodel>` with the respective model available in `seqsharp/pretrained_models` and use the corresponding simulated and empirical data collection (i.e. set of MSAs) available at (...).

### Arguments explained

**`--sim <str>`** example: `--sim my/path/to/simulated/data`

This argument specifies one or more directories containing files in fasta (*.fa*, *.fasta*) or phylib (*.phy*) format containing a simulated multiple sequence alignment (MSA). These data collections will be used to train, validate or test the classifier.

**`--emp <str>`** example: `--emp my/path/to/empirical/data`

With **`--emp`** you can specify the directory for empirical MSAs in fasta (.fa, .fasta) or phylib (.phy) format. 

**`-t`** or **`--train`**

This flag triggers the training of a model on the given data collections (specified by `--sim` and `--emp`). 
It will carry out a binary classification task to distinguish empirical and simulated MSAs. For this `--cfg` is required to configure the training.
Except if a pretrained model is specified (`--model`). Then, training of that model is resumed and the cfg.json file in the model folder is used.

**`--test`**

Using this flag a pretrained model is tested on one or more data collections (specified by `--sim` and `--emp`). The entire data collection is fed to all trained models for all folds. The performance is printed and saved to the folder of the pretrained models.

**`--validate`**

Here the given model is tested on validation data, i.e. there are k folds that are splits into validation and training data, for each fold there is a trained network that is now tested on the given validation data (note that there is a fixed random seed for splitting the data).

**`--attr`**

This functionality is currently not available.
With this flag attribution maps (saliency and integrated gradients) are computed and summarized for the validation data. It can be used with a given model or after training/resuming a model.

**`--clr`**

With this flag cyclic learning rates are used for training. For this a lr range must be specified in the config file.

**`-m <str>`** example: `-m my/path/to/my/pretrained/model`

This argument can be used to specify a directory containing one or more `.pth` files (one per fold). These files should contain a pretrained seq#-model. When resuming training of that model (`--train`) results will be stored in a new folder starting with *resume_*. It will be in the parent directory of the `-m` directory.

**`-c <str>`** example: `-c my/path/to/the/cfg/file.json`

Here you can input the configuration file in json format (an example is shown below). The parameters include parameters for: training (e.g. hyperparameters), the model architecture and parameters for processing the data (e.g. padding type).

**`-s <str>`** example: `-s my/path/to/where/models/and/results/will/be/stored`

Plots of learning curves, tables with model performances, trained models (.pth) and/or attribution maps are saved in this directory. More specifically they will be stored in a newly generated unique folder. 
Moreover, the config file, including e.g. learning rates determined during training for each fold will be saved to this folder. 

**`--shuffle`**

With this flag sites (columns) of the alignments will be permuted.

**`--ncpus`**

Number of CPUs to use (only applies for loading data).

### How to train and test a seq#-model

Train a model to distinguish real and simulated data:

`seqsharp -t --emp <path/to/emp/data> --sim <path/to/sim/data> -c <path/to/cfg.json> -s <path/to/results>
`

Resume training of a pretrained model: 

`seqsharp -t --emp <path/to/emp/data> --sim <path/to/sim1/data> -m <path/to/pretrained/models>
`

Use a pretrained model to test it on multiple data collections: 

`seqsharp --test --emp <path/to/emp/data> --sim <path/to/sim1/data> <path/to/sim2/data> -m <path/to/pretrained/models>
`

Test a pretrained model on simulated and empirical validation data: 

`seqsharp --validate --emp <path/to/emp/data> --sim <path/to/sim/data> -m <path/to/pretrained/models>
`

### cfg.json

An example config file looks like this:

```
{
    "data": 
    {
        "n_alignments": [7000, 8000],  # No. alignments
        "n_sites": "10000",  # No. sites i.e. sequence length
        "padding": "zeros",  # type of padding: "zeros", "gaps", or "data"
        "molecule_type": "protein"  # put "DNA" for nucleotides
    }, 
    "training": 
    {
        "batch_size": 128, 
        "epochs": 1000,  # max. No. epochs
        "lr": "",
        "lr_range": [1e-10,0.1],  # for learning rate range test
        "optimizer": "Adam",  # options: "Adam", "SGD", "Adagrad"
        "n_folds": 10  # the k in k-fold cross validation
    }, 
    "model": 
    {
        "channels": [21],  # No. of channels
        "n_lin_layer": 1,  # No. of linear layers
        "kernel_size": 1,  # kernel size per conv. layer
        "pooling": 0,  # 0: no pooling, 1: local max. pooling, 2: global avg. pooling
        "input_size": 1  # to use the max. No. of sites in your data collections remove this line 
    }, 
    "results_path": "",  # the path specified under -s will be put here automatically
    "comments": "Put your comments, notes here"}
```
##### Remarks to parameters in `data`:
- `n_alignments`: Here you can put the number of alignments you want to use. You can specify an Integer to use the same number for all data collections. To use all MSAs available you can specify `""`. Otherwise, the number of alignments for each input data collection needs to be specified (see example above) in the following order: empirical data collection, first simulated data collection (, second simulated data collection etc.).
- `n_sites`: All alignments with more than `n_sites` sites will not be considered. For an unlimited number of sites you can specify `""`.
- `padding`: Alignments with fewer than `input_size` sites will be padded on the edges, such that all MSAs have the same number of sites. Options are "zeros", "gaps" or "data". The latter will use uniformly sampled amino acids / nucleotides.

##### Remarks to parameters in `training`:
- `epochs`: If resuming training, this value should be the sum of the number of epochs already trained and the maximum number of epochs to be trained.
- `lr`: `lr` Can be either a learning rate e.g. *0.001* or a list of learning rates with one learning rate per fold e.g. *[1e-5, 0.01, 0.001, ...]* 
-  `lr_range`: If you specify `lr_range` as in the example file above the given range will be used to perform a learning rate range test. In this case `lr` will be ignored. For each fold a tenth of the determined upper bound will be used for training.

##### Remarks to parameters in `model`:
- `channels`: Determines the number of input/output channels and the number of convolution (conv.) layers. In the above example there are only 21 input channels specified, so no conv. layer will be used. The input channels stand for gaps and AAs (21) or nucleotides (5). To use conv. layers output channels need to be specified as well. E.g *[21, 210]*, means that there will be one conv. layer with 210 output channels or *[5, 100, 210]* means that 2 conv. layers will be used. The first of which will output 100 channels and the second 210. 
- `n_lin_layer`: Linear layers succeed the conv. layers. If more than one layer is to be used, then for each inner linear layer the number of output nodes is reduced by half and a ReLU is applied. The last linear layer for the binary classification has a single output node.
- `pooling`: Pooling succeeds the conv. layer(s). If multiple conv. layers are used, pooling will be applied after each layer in the case of maximum local pooling and solely after the last layer in the case of global average pooling.
- `input_size`: Corresponds to the sequence length, i.e. number of sites, plus padding. If it is `""`, it will be set to the maximum number of sites in all input MSAs. All MSAs with more than `input_size` sites will be cropped on the edges and all MSAs with fewer sites will be padded as explained above. In the example the input size of 1 indicates that the average MSA composition shall be used instead of site-wise compositions (hereby reducing the input size to 1). 

### Standard output of a successful run

When training a model (`--train`) the standard output should look similar to this:

```
Loading alignments ...
  9%|███████████▌                                                                                                                           | 599/6971 [00:05<00:53, 118.26it/s]


 => In 74 out of 600 MSAs 1.45% sites include ambiguous letters
Loaded 600 MSAs from 600 files from ../data/hogenom_fasta with success

  9%|████████████▍                                                                                                                           | 639/6971 [00:07<01:10, 90.13it/s]


Loaded 640 MSAs from 640 files from ../data/alisim_lg_gapless_trees with success

      hogenom_fasta              alisim_lg_gapless_trees             
           No.seqs.     No.sites                No.seqs.     No.sites
count    600.000000   600.000000              640.000000   640.000000
mean      24.863333   443.200000               25.107813   470.553125
std       33.109188   264.428448               35.035050   275.045787
min        4.000000    40.000000                4.000000    42.000000
25%        7.000000   237.500000                6.000000   243.750000
50%       13.000000   394.000000               12.000000   441.000000
75%       30.000000   593.250000               29.000000   644.250000
max      318.000000  1428.000000              271.000000  1400.000000
----------------------------------------------------------------
Generating alignment representations ...
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2/2 [00:01<00:00,  1.47it/s]

Compute device: cpu

Random seed: 42

----------------------------------------------------------------
        Batch size: 8
        Learning rate: [0.01, 0.01, 0.01]
----------------------------------------------------------------
FOLD 1
----------------------------------------------------------------
Epoch [0] Initial Model
Training: Loss: 0.6929, Acc.: 0.5
Validation: Loss: 0.6938, Acc.: 0.5, Emp. acc.: 0.0, Sim. acc.: 1.0
Epoch [1]
Training: Loss: 0.4922, Acc.: 0.7514
Validation: Loss: 0.5899, Acc.: 0.6475, Emp. acc.: 0.365, Sim. acc.: 0.9299
Epoch [2]
Training: Loss: 0.1984, Acc.: 0.9162
Validation: Loss: 0.6029, Acc.: 0.6783, Emp. acc.: 0.45, Sim. acc.: 0.9065
----------------------------------------------------------------
FOLD 2
----------------------------------------------------------------
Epoch [0] Initial Model
Training: Loss: 0.6926, Acc.: 0.5
Validation: Loss: 0.6928, Acc.: 0.5, Emp. acc.: 0.0, Sim. acc.: 1.0
Epoch [1]
Training: Loss: 0.4999, Acc.: 0.8428
Validation: Loss: 0.6082, Acc.: 0.7412, Emp. acc.: 0.905, Sim. acc.: 0.5775
Epoch [2]
Training: Loss: 0.1795, Acc.: 0.9646
Validation: Loss: 0.6249, Acc.: 0.7508, Emp. acc.: 0.835, Sim. acc.: 0.6667
----------------------------------------------------------------
FOLD 3
----------------------------------------------------------------
Epoch [0] Initial Model
Training: Loss: 0.6935, Acc.: 0.4813
Validation: Loss: 0.6931, Acc.: 0.5106, Emp. acc.: 0.81, Sim. acc.: 0.2113
Epoch [1]
Training: Loss: 0.6366, Acc.: 0.7978
Validation: Loss: 0.6605, Acc.: 0.7205, Emp. acc.: 0.61, Sim. acc.: 0.831
Epoch [2]
Training: Loss: 0.3336, Acc.: 0.9296
Validation: Loss: 0.5542, Acc.: 0.7599, Emp. acc.: 0.825, Sim. acc.: 0.6948

#########################  PERFORMANCE  #########################


---- Performance on validation data

        best_epoch      loss      bacc  acc_emp   acc_sim
Fold 1           2  0.602858  0.678271    0.450  0.906542
Fold 2           2  0.624862  0.750833    0.835  0.666667
Fold 3           2  0.554171  0.759918    0.825  0.694836

---- Summary

          loss      bacc  bacc_close_epochs   acc_emp   acc_sim  abs(emp-sim)
mean  0.593964  0.729674           0.603302  0.703333  0.756015      0.251680
std   0.036175  0.044747           0.141078  0.219450  0.131119      0.178439
min   0.554171  0.678271           0.503545  0.450000  0.666667      0.130164
max   0.624862  0.759918           0.703060  0.835000  0.906542      0.456542

Not saving models and evaluation plots. Please use --save and specify a directory if you want to save your results!
```

For testing a trained network (`--test`) you can expect a standard output similar to the following: 

```
Loading alignments ...
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 6971/6971 [00:59<00:00, 117.01it/s]


 => 2 file(s) have too few (<=10) sites after removing sites with ambiguous letters.
 => In 912 out of 6969 MSAs 1.34% sites include ambiguous letters
Loaded 6969 MSAs from 6971 files from ../data/hogenom_fasta with success

100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 6971/6971 [01:12<00:00, 96.64it/s]


Loaded 6971 MSAs from 6971 files from ../data/alisim_lg_gapless_trees with success

100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 6971/6971 [01:12<00:00, 95.97it/s]


Loaded 6971 MSAs from 6971 files from ../data/alisim_poisson_gapless_trees with success

100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 6971/6971 [01:12<00:00, 96.77it/s]


Loaded 6971 MSAs from 6971 files from ../data/alisim_lg_gc_gapless with success

      hogenom_fasta              alisim_lg_gapless_trees              alisim_poisson_gapless_trees              alisim_lg_gc_gapless             
           No.seqs.     No.sites                No.seqs.     No.sites                     No.seqs.     No.sites             No.seqs.     No.sites
count   6969.000000  6969.000000             6971.000000  6971.000000                  6971.000000  6971.000000          6971.000000  6971.000000
mean      24.462764   452.276223               24.470664   449.561756                    24.470664   451.254053            24.470664   451.472673
std       32.190250   275.759029               32.192759   274.914377                    32.192759   274.074809            32.192759   273.085064
min        4.000000    27.000000                4.000000    42.000000                     4.000000    41.000000             4.000000    41.000000
25%        7.000000   232.000000                7.000000   230.500000                     7.000000   233.500000             7.000000   232.000000
50%       13.000000   404.000000               13.000000   399.000000                    13.000000   403.000000            13.000000   398.000000
75%       29.000000   619.000000               29.000000   614.000000                    29.000000   619.500000            29.000000   622.000000
max      335.000000  1479.000000              335.000000  1478.000000                   335.000000  1477.000000           335.000000  1473.000000
----------------------------------------------------------------
Generating alignment representations ...
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 4/4 [00:30<00:00,  7.52s/it]


#########################  PERFORMANCE  #########################

              emp           alisim_lg_gapless_trees           alisim_poisson_gapless_trees           alisim_lg_gc_gapless          
             loss       acc                    loss       acc                         loss       acc                 loss       acc
Fold 1   0.211348  0.924953                0.199603  0.967006                     0.709262  0.867307             0.695358  0.742505
Fold 2   0.178218  0.938155                0.218943  0.959116                     0.839445  0.828145             0.815042  0.700473
Fold 3   0.189289  0.938729                0.217910  0.959690                     1.254817  0.488309             0.753810  0.712093
Fold 4   0.182957  0.934998                0.213677  0.962990                     1.329637  0.505953             0.829666  0.702195
Fold 5   0.256862  0.900703                0.160053  0.974609                     0.533980  0.900875             0.519630  0.814087
Fold 6   0.172813  0.941742                0.241159  0.954813                     1.209236  0.632908             0.890546  0.670205
Fold 7   0.160753  0.947912                0.252409  0.953235                     1.081125  0.742648             0.985645  0.644814
Fold 8   0.167361  0.943751                0.234418  0.956104                     1.258109  0.585999             0.923246  0.662172
Fold 9   0.189275  0.931841                0.204222  0.963994                     0.786245  0.846794             0.755206  0.723569
Fold 10  0.184705  0.933132                0.226271  0.959260                     1.133959  0.708937             0.861923  0.688567
min      0.160753  0.900703                0.160053  0.953235                     0.533980  0.488309             0.519630  0.644814
max      0.256862  0.947912                0.252409  0.974609                     1.329637  0.900875             0.985645  0.814087
std      0.027493  0.013264                0.025715  0.006368                     0.274843  0.151662             0.131825  0.047907
mean     0.189358  0.933592                0.216867  0.961082                     1.013582  0.710788             0.803007  0.706068
```
Please note that you will encounter the warning: `UserWarning: y_pred contains classes not in y_true` which is triggered if not all labels are present when using the balanced accuracy function of sklearn (which is the case here because it is applied to each class separately).

