# TimeSHAP

## AReM Dataset
Dataset taken from 
[this](https://archive.ics.uci.edu/ml/datasets/Activity+Recognition+system+based+on+Multisensor+data+fusion+(AReM)) link.

The used data on this dataset is the raw dataset with some exceptions:
 - `cycling\dataset9.csv` and `cycling\dataset14.csv` are unable to be read directly due to having a comma at the end - we remove the comma from these datasets and used them;
 - `sitting/dataset8.csv` only has 479 instances - we are ignoring this dataset;
 - We also excluded the bending datasets
 
##### Features
This dataset is composed of 6 numerical features:
 - avg_rss12;
 - var_rss12; 
 - avg_rss13; 
 - var_rss13; 
 - avg_rss23; 
 - var_rss23;

##### Sequences
After processing, we are left with 35520 total instances that represent
74 individual sequences of 480 elements each. 


 
 