# Simulating protein alignments

### simulate.py
This is a CLI to generate a set of multiple sequence alignments (data collection) with Alisim using parameters inferred from a set of protein MSAs from the HOGENOM database (available at TODO).

#### Requirements

1. Follow the installation steps in `seqsharp/README.md` under *Install*
2. To use the CLI please install iqtree2 from http://www.iqtree.org/
3. Run `iqtree2 --alisim` to verify that Alsim is working.
4. To use the UDM models by Schrempf et al.: `git clone  https://github.com/dschrempf/EDCluster.git` (optional)

#### Usage

1. Open a terminal and enter: `cd seqsharp/protein_simulations`
2. You can now run: `python simulate.py [args]`

Type `python simulate.py --help` to obtain the output below for possible arguments: 

```
usage: simulate.py [-h] -o OUTPATH [-t TREEPATH] [-n NSIM] [-m SUBM] [-g GAMMA] [--edclpath EDCLPATH] [-p {0004,0008,0016,0032,0064,0128,0192,0256,0512,1024,2048,4096}] [-i]
                   [--suffix SUFFIX]

optional arguments:
  -h, --help            show this help message and exit
  -o OUTPATH, --outpath OUTPATH
                        Path to directory of output simulations.
  -t TREEPATH, --treepath TREEPATH
                        Path to directory with phylogenetic tree files in Newick format.
  -n NSIM, --nsim NSIM  Number of alignments to simulate.
  -m SUBM, --subm SUBM  Substitution model as Alisim argument, e.g. "WAG" or "LG+C60"
  -g GAMMA, --gamma GAMMA
                        Gamma model, e.g. "G4" for four rate categories or "GC" for continuous Gamma distribution.
  --edclpath EDCLPATH   Path to EDCluster repository directory to use UDM models (for this clone https://github.com/dschrempf/EDCluster.git).
  -p {0004,0008,0016,0032,0064,0128,0192,0256,0512,1024,2048,4096}, --nprof {0004,0008,0016,0032,0064,0128,0192,0256,0512,1024,2048,4096}
                        Number of profiles of UDM model.
  -i, --indels          Simulate Indels using empirical Indel parameters.
  --suffix SUFFIX       Suffix for the simulation folder.
```

#### Reproduce simulated protein data collections

All simulated data collections and all phylogenies used for simulating can be found at TODO. 

To generate **alisim_poisson**, **alisim_wag**, **alisim_lg**, **alisim_lg_c60** the following command can be used replacing `<subm>` with *Poisson*, *WAG*, *LG* or *LG+C60* respectively.
```commandline
python simulate.py -o <path/to/output_dir> -t <path/to>/hogenom_trees_gapless --subm <subm>
```
For **alisim_lg_s0256**-simulations use the command:
```commandline
python simulate.py -o <path/to/output_dir> -t <path/to>/hogenom_trees_gapless --subm LG -p 0256 --edclpath <path/to>/EDCluster
```
For simulations with rate heterogeneity, **alisim_lg_s0256_g4** and **alisim_lg_s0256_gc**, run the following command replacing `<g>` by *G4* or *GC* respectively:
```commandline
python simulate.py -o <path/to/output_dir> -t <path/to>/hogenom_trees_gapless --subm LG -p 0256 --edclpath <path/to>/EDCluster -g <g>
```
For the simulations with Indels, **alisim_lg_s0256_gc_sabc**, run the following:
```commandline
python simulate.py -o <path/to/output_dir> -t <path/to>/hogenom_trees --subm LG -p 0256 --edclpath <path/to>/EDCluster -g GC --indels
```
Note that you need to use a different set of phylogenies, which are in *hogenom_trees*. These trees have been inferred from MSAs with Indels.
