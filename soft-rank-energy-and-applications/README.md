## Code for <a href="https://arxiv.org/abs/2111.00043/" style="color: blue; text-decoration: italic;text-decoration-style: dotted;">Multivariate soft rank via entropic optimal transport: sample efficiency and generative modeling</a>
This repository provides two applications of novel multivariate soft rank energy (sRE) and soft rank mmd (sRMMD). (a) Developing a generative model using sRE and sRMMD as the loss functions to produce MNIST-digits, (b) utilizing sRMMD as the loss in a deep generative model to produce valid knockoffs in order to select statistically significant features.
## Package Dependencies to sRMMMD-based knockoff filter
- python=3.6.5
- numpy=1.14.0
- scipy=1.0.0
- pytorch=0.4.1
- cvxpy=1.0.10
- cvxopt=1.2.0
- pandas=0.23.4
## How to run the code
1. To reproduce the MNIST results from the paper: <br>
    - Figure 1(b)- run 'mnist_figures_geneartion.py'<br>
    - Figure 1(a)- use lossType = 'mmd' and run 'mnist_figures_geneartion.py'<br>
    - Figure 1(c)- use lossType = 'sRMMD' and run 'mnist_figures_geneartion.py'<br>
2. To reproduce knockoff figures from the paper<br>
    - Extra package dependencies for other benchmarks <br>
        - DDLK : install the package from https://github.com/rajesh-lab/ddlk
        - KnockoffGAN : install Tensorflow v2 and use code from https://bitbucket.org/mvdschaar/mlforhealthlabpub/src/master/
    - Reproducing Figure 2(c)- run 'knockoff_figures_geneartion.py
    - Figure 2(a)- use distType = 'GaussianAR1' and run 'knockoff_figures_geneartion.py
    - Figure 2(b)- use distType = 'GaussianMixtureAR1' and run 'knockoff_figures_geneartion.py
    - Figure 2(d)- use distType = 'SparseGaussian' and run 'knockoff_figures_geneartion.py<br>
3. To reproduce Table 1 from the paper<br>
    - run real_dataset.py
  
     N:B: In case of any error regarding package dependices while running 'mnist_figures_geneartion.py' and 'real_dataset.py', run each method separately.
 
## Demo notebooks
These notebooks provide an overall view how sRMMD-knockoff filter works on synthetic and real data 
- [Examples/knockoff_synthetic_settings.ipynb](https://github.com/ShoaibBinMasud/soft-rank-energy-and-applications/blob/main/Examples/knockoff_synthetic_settings.ipynb) code to generate valid knockoffs using sRMMD.
- [Examples/knockoff_real_data.ipynb](https://github.com/ShoaibBinMasud/soft-rank-energy-and-applications/blob/main/Examples/knockoff_real_data.ipynb) metabolites selection using sRMMD knockoffs on the real data set available in [dataset/Real dataset](https://github.com/ShoaibBinMasud/soft-rank-energy-and-applications/tree/main/dataset/Real%20dataset)
