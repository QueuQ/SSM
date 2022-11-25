![SSM](https://github.com/QueuQ/SSM/tree/master/figures/pipeline_short_version.png)

[![License: CC BY 4.0](https://img.shields.io/badge/License-CC_BY_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)

This respository is based on the Continual Graph Learning Benchmark [CGLB](https://github.com/QueuQ/CGLB)

 <tr><td colspan="4"> <a href="#Get-Started">Get Started</a></td></tr> | <tr><td colspan="4"> <a href="#Dataset-Usages">Dataset Usages</a></td></tr> | <tr><td colspan="4"> <a href="#Pipeline-Usages">Pipeline Usages</a></td></tr> | <tr><td colspan="4"> <a href="#Evaluation-and-Visualization-Toolkit">Evaluation & Visualization Toolkit</a></td></tr> | <tr><td colspan="4"> <a href="#Benchmarks"> Benchmarks </a></td></tr> | <tr><td colspan="4"> <a href="#Acknowledgement"> Acknowledgement </a></td></tr>

 This is the official repository of Sparsified Subgraph Memory for Continual Graph Representation Learning (SSM), which was published in ICDM 2022.
 ```
 @inproceedings{zhang2022sparsified,
  title={Sparsified Subgraph Memory for Continual Graph Representation LearningGated Information Bottleneck for Generalization in Sequential Environments},
  author={Zhang, Xikun and Song, Dongjin and Tao, Dacheng},
  booktitle={2022 IEEE International Conference on Data Mining (ICDM)},
  year={2022},
  organization={IEEE}
}
 ```

 ## Get Started
 
To run the code, the following packages are required to be installed:
 
* python==3.7.10
* scipy==1.5.2
* numpy==1.19.1
* torch==1.7.1
* networkx==2.5
* scikit-learn~=0.23.2
* matplotlib==3.4.1
* ogb==1.3.1
* dgl==0.6.1

 For all experiments, the starting point is the ```train.py``` file, and the different configurations are assigned through the keyword arguments of the Argparse module. For example,
 
 ```
 python train.py --dataset Arxiv-CL \
       --backbone GCN \
       --gpu 0 \
       --epochs 200 \
       --sample_nbs False \
       --minibatch False \
       --repeats 5 \
       --ratio_valid_test 0.2 0.2 \
       --overwrite_result False \
       --perform_testing True
 ```

Since the graphs can be too large to be processed in one batch on most devices, the ```--minibatch``` argument could be specified to be ```True``` for training with the large graphs in mini-batches.
```
 python train.py --dataset Arxiv-CL \
       --backbone GCN \
       --gpu 0 \
       --epochs 200 \
       --sample_nbs False \
       --minibatch True \
       --batch_size 2000 \
       --repeats 5 \
       --ratio_valid_test 0.2 0.2 \
       --overwrite_result False \
       --perform_testing True
 ```
In the above example, besides specifying the ```--minibatch```, the size of each mini-batch is also specified through ```--batch_size```. Moreover, some graphs are extremely dense and will run out the memory even with mini-batch training, which could be addressed through the neighborhood sampling specified via ```--sample_nbs```. And the number of neighbors to sample for each hop is specified through ```--n_nbs_sample```.
There are also other customizable arguments, the full list of which can be found in ```train.py```.


### Modifying the train-validation-test Splitting

The splitting can be simply specified via the arguments when running the experiments. In our implemented pipeline, the corresponding arguments are the validation and testing ratios. For example,

```
 python train.py --dataset Arxiv-CL \
       --backbone GCN \
       --gpu 0 \
       --epochs 200 \
       --sample_nbs False \
       --minibatch True \
       --batch_size 2000 \
       --repeats 5 \
       --ratio_valid_test 0.4 0.4 \
       --overwrite_result False \
       --perform_testing True
 ```

The example above set the data ratio for validation and testing as 0.4 and 0.4, and the training ratio is automatically calculated as 0.2.
 
 ## Evaluation and Visualization Toolkit
 The introduction on the evaluation and visualization toolkit can be found in the official respository of CGLB [CGLB](https://github.com/QueuQ/CGLB).
 
 ## Acknowledgement
 The construction of this repository is based on [CGLB](https://github.com/QueuQ/CGLB).
 
