The official source code to Better Uncertainty Calibration via Proper Scores for Classification and Beyond (NeurIPS'22)

https://arxiv.org/abs/2203.07835

To reproduce the results, do the following.

## Classification (Figure 1, 2, 3, 5, 6, and 7)

This section is exclusively about the classification calibration experiments.

### Logits

The logits are pretrained from `https://github.com/markus93/NN_calibration/tree/master/logits` and `https://github.com/AmirooR/IntraOrderPreservingCalibration`.
For quality of life and backup redundancy, we also provide them in the following Google Drive link https://drive.google.com/drive/folders/10XVg_anBCWmjzjh_Hb-A7GYcgjHLypax?usp=sharing

### Running the experiments (Figure 1, 3, 5, 6, and 7)

Execute `bash run_experiments.sh`.
This will take several days if the used CPU has only few cores.
Lower the parameter `start_rep` to lower runtime.
This will increase the variance of the results.
To make the results more stable, the command can be executed multiple times (results are appended instead of overwritten).
Indeed, we ran it twice.
Upon request, we can provide our result files.
The seeds are set automatically, since we set the samples to a size where the seed should
not matter anymore.
Setting a seed manually is supported for reproducibility, even though this is against
the nature of the experiments.
All results will be stored in the `results/` folder.

### Plotting (Figure 1, 3, 5, 6, and 7)

There are two options:
To receive all the plots in the paper (and even more), execute
`python plotting.py`.
This takes a while (3-6 minutes) and you may receive errors if certain result files are not found.
Contrary, running the notebooks allows producing and inspecting each plot individually.

### ECE estimation of simulated models (Figure 2)

Open and run the jupyter notebook `ECE bias ground truth simulation.ipynb`.
The plot is directly displayed in the last cell.
The simulation is light-weight and can be finished quickly on a typical laptop.
Indeed, we ran the simulation on an M1 MacBook Pro 2021 in minutes.


## Variance regression (Figure 4, 8, and 9)

This section is exclusively about the regression calibration experiments.
Note, that the DSS score is called PMCC in the code due to technical debt.
Ideally, this is fixed some time in the future.
Further, we used major parts of the code from Wiedman et al 2021 (`https://github.com/devmotion/Calibration_ICLR2021`).
But, this also means the regression experiments are done in Julia and require new dependencies being installed compared to the classification experiments.

Again, we ran the experiments on an M1 MacBook Pro 2021, but a run may require up to an hour.

### Residential Building (Figure 4 and 8)

First, download the dataset from `http://archive.ics.uci.edu/ml/datasets/Residential+Building+Data+Set` and place the csv file into `data/`.
Then, open and run the jupyter notebook `recalibration_friedman1.ipynb`.
To plot the figures, open the jupyter notebook `plotting_recal_regr.ipynb` and set the variable `task` to `ResBuild`, then run all the cells.
The plots are directly shown in the notebook.

### Friedman 1 (Figure 9)

Open and run the jupyter notebook `recalibration_friedman1.ipynb`.
To plot the figures, open the jupyter notebook `plotting_recal_regr.ipynb` and set the variable `task` to `friedman1_var`, then run all the cells.
The plots are directly shown in the notebook.
