# TimeSHAP
TimeSHAP is a model-agnostic, recurrent explainer that builds upon KernelSHAP and 
extends it to the sequential domain.
TimeSHAP computes event/timestamp- feature-, and cell-level attributions. 
As sequences can be arbitrarily long, TimeSHAP also implements a pruning algorithm
based on Shapley Values, that finds a subset of consecutive, recent events that contribute
the most to the decision.


This repository is the code implementation of the TimeSHAP algorithm 
present in the paper `TimeSHAP: Explaining Recurrent Models through Sequence Perturbations`
published in KDD 2021.

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

## TimeSHAP in 30 seconds

#### Inputs
- Model being explained;
- Instance(s) to explain;
- Background instance.

#### Outputs
- Local pruning output; (explaining a single instance)
- Local event explanations; (explaining a single instance)
- Local feature explanations; (explaining a single instance)
- Global pruning statistics; (explaining multiple instances)
- Global event explanations; (explaining multiple instances)
- Global feature explanations; (explaining multiple instances)

### TImeSHAP Explanation Methods
TimeSHAP offers several methods to use depending on the desired explanations.
Local methods provide detailed view of a model decision corresponding
to a specific sequence being explained.
Global methods aggregate local explanations of a given dataset
to present a global view of the model.

#### Local Explanations
##### Pruning

[`local_pruning()`](src/timeshap/explainer/pruning.py) performs the pruning
algorithm on a given sequence with a given user defined tolerance and returns 
the pruning index along the information for plotting.

[`plot_temp_coalition_pruning()`](src/timeshap/plot/pruning.py) plots the pruning 
algorithm information calculated by `local_pruning()`.

<img src="resources/images/pruning.png" width="100">

##### Event level explanations

[`local_event()`](src/timeshap/explainer/event_level.py) calculates event level explanations
of a given sequence with the user-given parameteres and returns the respective 
event-level explanations.

[`plot_event_heatmap()`](src/timeshap/plot/event_level.py) plots the event-level explanations
calculated by `local_event()`.

<img src="resources/images/event_level.png" width="100">

##### Feature level explanations

[`local_feat()`](src/timeshap/explainer/feature_level.py) calculates feature level explanations
of a given sequence with the user-given parameteres and returns the respective 
feature-level explanations.

[`plot_feat_barplot()`](src/timeshap/plot/feature_level.py) plots the feature-level explanations
calculated by `local_feat()`.

<img src="resources/images/feature_level.png" width="100">

##### Cell level explanations

[`local_cell_level()`](src/timeshap/explainer/cell_level.py) calculates cell level explanations
of a given sequence with the respective event- and feature-level explanations
and user-given parameteres, returing the respective cell-level explanations.

[`plot_cell_level()`](src/timeshap/plot/cell_level.py) plots the feature-level explanations
calculated by  `local_cell_level()`.

<img src="resources/images/cell_level.png" width="100">

##### Local Report

[`local_report()`](src/timeshap/explainer/local_methods.py) calculates TimeSHAP 
local explanations for a given sequence and plots them.

<img src="resources/images/local_report.png" width="600">

#### Global Explanations


##### Global pruning statistics

[`prune_all()`](src/timeshap/explainer/pruning.py) performs the pruning
algorithm on multiple given sequences.

[`pruning_statistics()`](src/timeshap/plot/pruning.py) calculates the pruning
statistics for several user-given pruning tolerances using the pruning
data calculated by `prune_all()`, returning a `pandas.DataFrame` with the statistics.


##### Global event level explanations

[`event_explain_all()`](src/timeshap/explainer/event_level.py) calculates TimeSHAP 
event level explanations for multiple instances given user defined parameters.

[`plot_global_event()`](src/timeshap/plot/event_level.py) plots the global event-level explanations
calculated by `event_explain_all()`.

<img src="resources/images/global_event.png" width="100">

##### Global feature level explanations

[`feat_explain_all()`](src/timeshap/explainer/feature_level.py) calculates TimeSHAP 
feature level explanations for multiple instances given user defined parameters.

[`plot_global_feat()`](src/timeshap/plot/feature_level.py) plots the global feature-level 
explanations calculated by `feat_explain_all()`.

<img src="resources/images/global_feat.png" width="100">


##### Global report
[`global_report()`](src/timeshap/explainer/global_methods.py) calculates TimeSHAP 
explanations for multiple instances, aggregating the explanations on two plots
and returning them.

<img src="resources/images/global_report.png" width="400">



## Tutorial
In order to demonstrate TimeSHAP interfaces and methods, you can consult
[AReM.ipynb](notebooks/AReM/AReM.ipynb). 
In this tutorial we get an open-source dataset, process it, train 
Pytorch recurrent model with it and use TimeSHAP to explain it, showcasing all 
previously described methods.

Additionally, we also train a TensorFlow model on the same dataset 
[AReM_TF.ipynb](notebooks/AReM/AReM_TF.ipynb).

## Repository Structure

- [`notebooks`](notebooks) - tutorial notebooks demonstrating the package;
- [`src/timeshap`](src/timeshap) - the package source code;
  - [`src/timeshap/explainer`](src/timeshap/explainer) - TimeSHAP methods to produce the explanations
  - [`src/timeshap/explainer/kernel`](src/timeshap/explainer/kernel) - TimeSHAPKernel
  - [`src/timeshap/plot`](src/timeshap/plot) - TimeSHAP methods to produce explanation plots
  - [`src/timeshap/utils`](src/timeshap/utils) - util methods for TimeSHAP execution
  - [`src/timeshap/wrappers`](src/timeshap/wrappers) - Wrapper classes for models in order to ease TimeSHAP explanations

## Citing TimeSHAP
```
@inproceedings{10.1145/3447548.3467166, #bento2021timeshap
    author = {Bento, Jo\~{a}o and Saleiro, Pedro and Cruz, Andr\'{e} F. and Figueiredo, M\'{a}rio A.T. and Bizarro, Pedro},
    title = {TimeSHAP: Explaining Recurrent Models through Sequence Perturbations},
    year = {2021},
    isbn = {9781450383325},
    publisher = {Association for Computing Machinery},
    address = {New York, NY, USA},
    url = {https://doi.org/10.1145/3447548.3467166},
    doi = {10.1145/3447548.3467166},
    booktitle = {Proceedings of the 27th ACM SIGKDD Conference on Knowledge Discovery & Data Mining},
    pages = {2565–2573},
    numpages = {9},
    keywords = {SHAP, Shapley values, TimeSHAP, XAI, RNN, explainability},
    location = {Virtual Event, Singapore},
    series = {KDD '21}
}
```