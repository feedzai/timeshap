# TimeSHAP
TimeSHAP is a model-agnostic recurrent explainer that builds upon KernelSHAP and extends it to the sequential domain.
TimeSHAP computes feature-, timestep-, and cell-level attributions. 

This repository is the code implementation of the TimeSHAP algorithm 
present in the paper `TimeSHAP: Explaining Recurrent Models through Sequence Perturbations`
published in KDD 2022.

## Install TimeSHAP
Clone the repository into a local directory using:
```
git clone https://github.com/feedzai/timeshap.git
```

Move into TimeSHAP directory and install the package using pip:

```
pip install timeshap .
```

To test your installation, start a Python session in your terminal using

```
python
```

And import TimeSHAP

```
import timeshap
```

## Tutorial
In order to demonstrate TimeSHAP interfaces and methods, you can consult
![AReM.ipynb](notebooks/AReM/AReM.ipynb) where we get a raw dataset, train a 
recurrent model and use TimeSHAP to explain ir.

## TimeSHAP explanation Plots

### Global Explanations

##### Global Event Explanations
![](resources/images/global_event.png)

##### Global Feature Explanations
![](resources/images/global_feat.png)


### Local Explanations

##### Pruning Algorithm
![](resources/images/pruning.png)

##### Event Explanations
![](resources/images/event_level.png)

##### Feature Explanations
![](resources/images/feature_level.png)

##### Cell Explanations
![](resources/images/cell_level.png)

## Repository Structure

- [`notebooks`](notebooks) - tutorial notebooks demonstrating the package;
- [`src/timeshap`](src/timeshap) - the package source code;
  - [`src/timeshap/kernel`](src/timeshap/kernel) - original [shap](https://github.com/slundberg/shap) files altered to implement TimeSHAP; 
  - [`src/timeshap/methods`](src/timeshap/methods) - TimeSHAP methods to produce the explanations
  - [`src/timeshap/plot`](src/timeshap/plot) - TimeSHAP methods to produce explanation plots
  - [`src/timeshap/utils`](src/timeshap/utils) - util methods for TimeSHAP execution
  - [`src/timeshap/wrappers`](src/timeshap/wrappers) - Wrapper classes for models in order to ease TimeSHAP explanations

## Citing TimeSHAP
```
@inproceedings{DBLP:conf/kdd/0002SCFB21,
  author    = {Jo{\~{a}}o Bento and Pedro Saleiro and Andr{\'{e}} Ferreira Cruz 
               and M{\'{a}}rio A. T. Figueiredo and Pedro Bizarro},
  editor    = {Feida Zhu and Beng Chin Ooi and Chunyan Miao},
  title     = {TimeSHAP: Explaining Recurrent Models through Sequence Perturbations},
  booktitle = {{KDD} '21: The 27th {ACM} {SIGKDD} Conference on Knowledge Discovery
               and Data Mining, Virtual Event, Singapore, August 14-18, 2021},
  pages     = {2565--2573},
  publisher = {{ACM}},
  year      = {2021},
  url       = {https://doi.org/10.1145/3447548.3467166},
  doi       = {10.1145/3447548.3467166},
  timestamp = {Sun, 23 Jan 2022 17:18:03 +0100},
  biburl    = {https://dblp.org/rec/conf/kdd/0002SCFB21.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```