# Better Uncertainty Calibration via Proper Scores for Classification and Beyond

The official source code to [Better Uncertainty Calibration via Proper Scores for Classification and Beyond (NeurIPS'22)](https://arxiv.org/abs/2203.07835).

Also available on [OpenReview](https://openreview.net/forum?id=PikKk2lF6P).

The classification experiments are all done in Python, while the variance regression ones are in Julia.
This is because the regression case is built upon [Wiedmann et al 2021](https://github.com/devmotion/Calibration_ICLR2021).
We split up the (install) description of the figures in these two categories: Classification (ECE simulation, CIFAR-10/100, ImageNet), and variance regression (extended Friedman 1, Residiential Housing).
This way, you do not have to install Julia if you are only interested in the classification case, or vice versa.

## Classification

### Install

After cloning this repository, create a conda environment via the provided yaml file.
For this, install Anaconda and run `conda env create -f condaenv.yml`
(like [here](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html)).
Then, activate the enironment with `conda activate unc_cal`.

### ECE estimation of simulated models (Figure 2)

Open and run the jupyter notebook `ECE bias ground truth simulation.ipynb`.
The plot is directly displayed in the last cell.
The simulation is light-weight and can be finished quickly on a typical laptop.
Indeed, we ran the simulation on an M1 MacBook Pro 2021 in minutes.


### Real-world data (Figure 1, 3, 5, 6, and 7)

This section is about the real-world classification calibration experiments.

#### Logits

The logits are pretrained from [Kull et al 2019](https://github.com/markus93/NN_calibration/tree/master/logits) and [Rahimi et al 2020](https://github.com/AmirooR/IntraOrderPreservingCalibration).
For quality of life and backup redundancy, we also provide them in [this Google Drive folder](https://drive.google.com/drive/folders/10XVg_anBCWmjzjh_Hb-A7GYcgjHLypax?usp=sharing). To download the folder, simply run `gdown https://drive.google.com/drive/u/1/folders/10XVg_anBCWmjzjh_Hb-A7GYcgjHLypax -O logits --folder`.

#### Running the experiments (Figure 1, 3, 5, 6, and 7)

All results will be located in the `results/` folder.
To run all of the experiments, execute `bash run_experiments.sh`.
This will take several days if the used CPU has only few cores.
Lower the parameter `start_rep` to lower runtime.
This will increase the variance of the results.
The command can also be executed multiple times (results are appended instead of overwritten; we ran it twice).
The seeds are set according to the local time and differ for each rerun.
The sample size should be large enough such that the seed does not matter.
Setting seeds manually is supported and ours are stored in our result files.
Since the runtime can be infeasible for some CPUs, we also provide our results files in [this Google drive folder](https://drive.google.com/drive/folders/1pP3RhgIdTXpKLArmiyECTo94VcAf0zll?usp=sharing).
To download the folder, simply run `gdown https://drive.google.com/drive/folders/1pP3RhgIdTXpKLArmiyECTo94VcAf0zll -O results --folder`.


#### Plotting (Figure 1, 3, 5, 6, and 7)

There are two options:
To receive all the plots in the paper (and even more), execute
`python plotting.py`.
This takes a while (3-6 minutes).
Contrary, running the notebooks allows producing and inspecting each plot individually.


## Variance regression (Figure 4, 8, and 9)

This section is exclusively about the regression calibration experiments.
Note, that the DSS score is called PMCC in the code due to technical debt.
Ideally, this is fixed some time in the future.
Further, we used major parts of the code from [Wiedmann et al 2021](https://github.com/devmotion/Calibration_ICLR2021).
But, this also means the regression experiments are done in Julia and require new dependencies being installed compared to the classification experiments.

We used the same dependencies as [Wiedmann et al 2021](https://github.com/devmotion/Calibration_ICLR2021) and the following instructions are copied from there.

(Start of copy)

### Install Julia 

The experiments were performed with Julia 1.5.3. You can download the official binaries from
the [Julia webpage](https://julialang.org/downloads/).

For increased reproducibility, the `nix` environment in this folder provides a fixed Julia
binary:
1. Install [nix](https://github.com/NixOS/nix#installation).
2. Navigate to this folder and activate the environment by running
   ```shell
   nix-shell
   ```
   in a terminal.
 
(End of copy)

Again, we ran the experiments on an M1 MacBook Pro 2021, where a run required up to an hour.


### Residential Building (Figure 4 and 8)

First, download the dataset from `http://archive.ics.uci.edu/ml/datasets/Residential+Building+Data+Set` and place the csv file into `data/`.
Then, open and run the jupyter notebook `recalibration_friedman1.ipynb`.
To plot the figures, open the jupyter notebook `plotting_recal_regr.ipynb` and set the variable `task` to `ResBuild`, then run all the cells.
The plots are directly shown in the notebook.

### Friedman 1 (Figure 9)

Open and run the jupyter notebook `recalibration_friedman1.ipynb`.
To plot the figures, open the jupyter notebook `plotting_recal_regr.ipynb` and set the variable `task` to `friedman1_var`, then run all the cells.
The plots are directly shown in the notebook.


## Citation

If you found this code useful, please cite our paper
```
@inproceedings{
   gruber2022better,
   title={Better Uncertainty Calibration via Proper Scores for Classification and Beyond},
   author={Sebastian Gregor Gruber and Florian Buettner},
   booktitle={Advances in Neural Information Processing Systems},
   editor={Alice H. Oh and Alekh Agarwal and Danielle Belgrave and Kyunghyun Cho},
   year={2022},
   url={https://openreview.net/forum?id=PikKk2lF6P}
}
```
