# NN-verification

## Intro
This responsitory is the code for project in the course [ECE285,UCSD](https://zhengy09.github.io/ECE285/project.html).
It is mainly based on material in Wong and Kolter (2018); Raghunathan et al.
(2018b); Fazlyab et al. (2020); Batten et al. (2021). We try to illustrate the methodology, provide
further exposition and connections between those works, and reconstruct the methods and show
them in examples.

## Contents
The overall code and input data is from the codebase of Raghunathan et al.
The addition is:
- code/auto_certify.py: auto compute the optimal value for many test examples.
- code/matlab_lp.m: matlab implementation of LP relaxation
- code/matlab_SDP3.m: Matlab implementation of SDP3 and SDP4 relaxation
- review_verification_for_NN.pdf: review report

## How to use:
```python
python certify.py --checkpoint ../models/{network}.ckpt --model_json ../model_details/{network}.json 
--test_input ../mnist_permuted_data/test-0.npy --true_class 8 --adv_class 0 --epsilon 0.1 --matlab_folder matlab
```
network could be {nips_lp, nips_sdp, nips_pdg}

## package
You should have mosek in your matlab and tensorflow.  
Feel free to contact me if you have trouble running the algorithm:D.
