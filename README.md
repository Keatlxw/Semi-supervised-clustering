# Semi-Supervised Clustering Awesome Repository

[![Awesome](https://awesome.re/badge.svg)](https://awesome.re)

This repository contains a list of resources related to semi-supervised clustering. Semi-supervised clustering is a type of clustering where some labels (or a partial set) are available, in addition to the data. The goal is to leverage these labels, along with the data, to perform better clustering. The problem is not well understood, and there are a variety of approaches to solve semi-supervised clustering.

Contributors and new resources are welcome. Please make a pull request or open an issue.

## Papers

### Survey Papers

- [N. Srivastava, M. Ahmed, and G. Hulten. A Primer on Semi-Supervised Clustering. In Proceedings of Workshop on Clustering High Dimensional Data and its Applications at SIAM International Conference on Data Mining (SDM16).](http://www1.se.cuhk.edu.hk/~mianshui/pub/sdm2016ws.pdf)
# Semi-Supervised Clustering Awesome Repository

[![Awesome](https://awesome.re/badge.svg)](https://awesome.re)

This repository contains a list of resources related to semi-supervised clustering. Semi-supervised clustering is a type of clustering where some labels (or a partial set) are available, in addition to the data. The goal is to leverage these labels, along with the data, to perform better clustering. The problem is not well understood, and there are a variety of approaches to solve semi-supervised clustering.

Contributors and new resources are welcome. Please make a pull request or open an issue.

## Papers

### Survey Papers

- [N. Srivastava, M. Ahmed, and G. Hulten. A Primer on Semi-Supervised Clustering. In Proceedings of Workshop on Clustering High Dimensional Data and its Applications at SIAM International Conference on Data Mining (SDM16).](http://www1.se.cuhk.edu.hk/~mianshui/pub/sdm2016ws.pdf)

### Algorithms

#### Ensemble Methods
 - [S. Laine and T. Aila. Temporal Ensembling for Semi-Supervised Learning. International Conference on Learning Representations (ICLR), 2017.](https://openreview.net/pdf?id=B1Yy1BxCZ)
#### Graph-based Methods

- Laplacian Regularized K-means
    - [X. Zhu, Z. Ghahramani, and J. Lafferty. Semi-Supervised Learning Using Gaussian Fields and Harmonic Functions. In Proceedings of the 20th International Conference on Machine Learning (ICML’03), pp. 912–919.](http://proceedings.mlr.press/v28/zhu03a.pdf)
    - [O. N. Bedreag and M. Saupe. Robust Semi-Supervised Clustering with Constraint Propagation for Large-Scale Facial Image Sets. International Journal of Computer Vision, 105(3), pp. 227–240, 2013.](https://link.springer.com/content/pdf/10.1007%2Fs11263-013-0624-x.pdf)
    - [C. Gao, S. Zhang, Y. Ning, Y. Cui and Z. Huang. Laplacian Regularized Soft K-Means Clustering for Fuzzy Image Segmentation. Fuzzy Systems, IEEE Transactions on, DOI: 10.1109/TFUZZ.2015.2407274, 2015.](https://ieeexplore.ieee.org/abstract/document/7131068)
    - [Y. Shi, Y. Liu, and Y. Guo. Laplacian Regularized Weighted K-means for Image Clustering. Journal of Visual Communication and Image Representation, 25(1), pp. 78–85, 2014.](https://www.researchgate.net/profile/Yang-Shi-50/publication/255599987_Laplacian_Regularized_Weighted_K-means_for_Image_Clustering/links/0c96053944a029fce8000000.pdf)
    - [A. F. Alenezi and J. Xiong. A New Laplacian Regularized K-means Algorithm for Hyperspectral Image Clustering. PLoS ONE, 10(6): e0126126, 2015.](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4445308/pdf/pone.0126126.pdf)
- Spectral Clustering
    - [U. von Luxburg. A Tutorial on Spectral Clustering. Statistics and Computing, 17(4), pp. 395–416, 2007.](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.165.9323&rep=rep1&type=pdf)
    - [Z. Wang and Q. Wang. Spectral Clustering with Iterative Refinement Using Label Information. Pattern Recognition Letters, 34(13), pp. 1520–1528, 2013.](http://www.sciencedirect.com/science/article/pii/S0167865513001166)
    - []()
- Graph-based K-means
    - [S. E. Ihler, J. W. Fisher III, and A. Willsky. Loopy belief propagation: convergence and effects of message errors. Journal of Machine Learning Research, 6, pp. 905–935, 2005.](http://www.jmlr.org/papers/volume6/ihler05a/ihler05a.pdf)
    - [G. Zhou, Z. Chen, and X. Gao. Graph-Based K-Means Clustering. Journal of Computational Information Systems, 9(12), pp. 4637–4644, 2013.](http://www.jofcis.com/UploadFiles/201312/2013120917223930.pdf)
#### Neural Network-based Methods

- [J. Yang, D. Parikh, and D. Batra. Joint Unsupervised Learning of Deep Representations and Image Clusters. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR'16), pp. 5147-5156.](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Yang_Joint_Unsupervised_Learning_CVPR_2016_paper.pdf)

#### Other Methods
   - [S. Laine and T. Aila. Temporal Ensembling for Semi-Supervised Learning. International Conference on Learning Representations (ICLR), 2017.](https://openreview.net/pdf?id=B1Yy1BxCZ)

### Datasets

- [A. K. Jain. Data Clustering: 50 Years Beyond K-means. Handbook of Cluster Analysis, pp. 59-70, 2015.](https://link.springer.com/chapter/10.1007/978-3-319-13936-0_5)
- [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/index.php)
- [Semi-Supervised Classification and Clustering (SSCC) Repository](http://cs.joensuu.fi/sipu/datasets/)
- [The Rat Genome Database (RGD)](https://rgd.mcw.edu/)
- [UNSW-NB15 Network Intrusion Detection Dataset](https://www.unsw.adfa.edu.au/unsw-canberra-cyber/cybersecurity/ADFA-NB15-Datasets/)
- [CURE-TSD: Toronto Faces Dataset](https://www.cs.toronto.edu/~jepson/csc320/)
- [20 Newsgroups Text Classification](http://qwone.com/~jason/20Newsgroups/)
- [MNIST Dataset](http://yann.lecun.com/exdb/mnist/)

## Code

### Python

- [scikit-learn](https://scikit-learn.org/stable/modules/clustering.html#semi-supervised-clustering)
- [scikit-multilearn](http://scikit.ml/api/skmultilearn.cluster.cobras.html)
- [spectral_clustering.py](https://github.com/harp/blob/master/algorithms/spectral_clustering.py)
- [sskmeans.py](https://github.com/pmtamayo/sskmeans)
- [sskmeans-scikit](https://github.com/timitsie/sskmeans_scikit)
- [higashi-cluster](https://github.com/iHilmi/higashi-cluster)
- [clustering-with-labels](https://github.com/MihaiBuda/Clustering-With-Labels)
- [kgcnn](https://github.com/ailabstw/kgcnn/tree/master/clustering/y-clustering)
- [semisup-semi-supervised-learning-for-python](https://github.com/tmadl/semisup)
- [RB-Sklearn](https://github.com/tmadl/RB-sklearn)
- [lssc_clustering](https://github.com/amoussavi/lssc_clustering)

### R

- [ssClust](https://www.rdocumentation.org/packages/ssClust/versions/0.1.3)

## Contribute

Contributions are always welcome! Please read the [contribution guidelines](CONTRIBUTING.md) first.

## License

This repository is licensed under the [MIT License](LICENSE).
