# Attention based Sequence to Sequence for RUL
 Proposed ATS2S model on C-MAPSS dataset.

This repo contains code of the paper, 	[ Attention based Sequence to Sequence for RUL] has been accepted in Neurocomputing, 2021. It includes code for estimating remaining useful life machine using sequence to sequence model with axuulary task to improve the feature representation.

### Dependencies
This code requires the following:
* python 2.\* or python 3.\*
* Pytorch v1.2+

### Data
The model performance is tested on NASA turbofan engines dataset [https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/#turbofan]. 

### Data Preprocessing
After downloading you can do the data preprocessing steps you can check this file `data_processing.py`

### Usage
To run the code, we have two main models, single working condition model `model_single_wk_data.py` and multiple working model  `model_multi_wk_data.py. The files will show the training results and then print the performance on test set. 

