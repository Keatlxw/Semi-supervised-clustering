[python-img]: https://img.shields.io/github/languages/top/Keatlxw/Semi-supervised-clustering?color=lightgrey
[stars-img]: https://img.shields.io/github/stars/Keatlxw/Semi-supervised-clustering?color=yellow
[stars-url]: https://github.com/Keatlxw/Semi-supervised-clustering/stargazers
[fork-img]: https://img.shields.io/github/forks/Keatlxw/Semi-supervised-clustering?color=lightblue&label=fork
[fork-url]: https://github.com/Keatlxw/Semi-supervised-clustering/network/members
[visitors-img]: https://visitor-badge.glitch.me/badge?page_id=Keatlxw.Semi-supervised-clustering
[adgc-url]: https://github.com/Keatlxw/Semi-supervised-clustering

# Semi-Supervised Clustering Awesome Repository

[![Awesome](https://awesome.re/badge.svg)](https://awesome.re)

This repository contains a list of resources related to semi-supervised clustering. Semi-supervised clustering is a type of clustering where some labels (or a partial set) are available, in addition to the data. The goal is to leverage these labels, along with the data, to perform better clustering. The problem is not well understood, and there are a variety of approaches to solve semi-supervised clustering.

Contributors and new resources are welcome. Please make a pull request or open an issue.

[![Made with Python][python-img]][adgc-url]
[![GitHub stars][stars-img]][stars-url]
[![GitHub forks][fork-img]][fork-url]
[![visitors][visitors-img]][adgc-url]

--------------

# What's Semi-Supervised Clustering？
Semi-supervised clustering refers to a type of clustering algorithm where the clustering process is partially guided by available labeled data in addition to unlabeled data. In semi-supervised clustering, the goal is to leverage the limited labeled data, along with the unlabeled data, to perform better clustering than what could be achieved by using only the unlabeled data.

In the context of semi-supervised clustering, the available labeled data may consist of a small but representative sample of the data set, or a partial set of the true cluster labels. The unlabeled data set, on the other.More details can be found in the survey paper.[Link](https://www.sciencedirect.com/science/article/abs/pii/S0020025523002840)




## Important Survey Papers

| Year | Title | Venue | Paper | Code |
|------|-------|-------|-------|------|
| 2021 | Semi-Supervised Clustering: A Study on User-Guided and Active Approaches | IEEE Transactions on Neural Networks and Learning Systems | [Link](https://ieeexplore.ieee.org/abstract/document/9261803) | [Link](https://github.com/userguidedsemiSEMI) |
| 2020 | MixMatch: A Holistic Approach to Semi-Supervised Learning | NeurIPS 2019 | [Link](https://proceedings.neurips.cc/paper/2019/hash/1b5e3ef18d27ab1e91c3c2b99a5477f8-Abstract.html) | [Link](https://github.com/google-research/mixmatch)|
| 2019 | Deep Co-Clustering for Unsupervised and Semi-Supervised Learning | ICDM 2018 | [Link](https://ieeexplore.ieee.org/abstract/document/8595135) | - |
| 2018 | A Review of Semi-Supervised Clustering | IEEE Transactions on Knowledge and Data Engineering | [Link](https://ieeexplore.ieee.org/abstract/document/8428408) | - |
| 2017 | Semi-Supervised Clustering with Intercluster Discrimination | IEEE Transactions on Pattern Analysis and Machine Intelligence | [Link](https://ieeexplore.ieee.org/abstract/document/7962925) | - | 

## Papers
### Co-training based clustering
| Year | Title | Venue | Paper | Code |
|------|-------|-------|-------|------|
2023 | Co-Training-Based Outlier Detection for IoT Data Streams | IEEE/ACM Transactions on Networking | [Link](https://ieeexplore.ieee.org/document/9565368) | N/A
2022 | Multi-Modal Co-Training for Clustering | International Joint Conference on Neural Networks | [Link](https://ieeexplore.ieee.org/document/9662443) | N/A
2021 | Co-Training Multi-view Clustering with Enhanced Local-Global Consistency and Orthogonality | Neurocomputing | [Link](https://www.sciencedirect.com/science/article/pii/S0925231221010426) | N/A
| 2020 | Regularized Co-Training for Mapping Unlabeled Data between Different Domains | IEEE Transactions on Neural Networks and Learning Systems | [Link](https://ieeexplore.ieee.org/abstract/document/9096188) | - |
| 2019 | Co-Training Embeddings of Knowledge Graphs and Entity Descriptions for Cross-lingual Entity Alignment | ACL 2019 | [Link](https://www.aclweb.org/anthology/P19-1546/) | - |
2018 | Co-training over Unlabeled Data with Domain-Specific Information for Multi-domain Sentiment Analysis | IEEE International Symposium on Signal Processing and Information Technology | [Link](https://ieeexplore.ieee.org/document/8468805) | N/A
2017 | Co-Training Deep Convolutional Networks for Semi-Supervised Clustering | IEEE Transactions on Multimedia | [Link](https://ieeexplore.ieee.org/document/7838145) | [Code](https://github.com/Yangyangii/CoNet)
2016 | Multi-view Co-training for Semi-supervised Clustering | International Conference on Multimedia Modeling | [Link](https://link.springer.com/chapter/10.1007/978-3-319-27671-7_10) | N/A
2015 | Co-Training Ensemble Clustering for High-Dimensional Dat | Symposium on Applied Computing | [Link](https://dl.acm.org/doi/abs/10.1145/2695664.2695878) | N/A
| 2015 | A Co-Training Approach for Multi-View Spectral Clustering | IEEE Transactions on Image Processing | [Link](https://ieeexplore.ieee.org/abstract/document/7040050) | - |
| 2014 | Co-training for domain adaptation of sentiment classifiers | EMNLP 2014 | [Link](https://www.aclweb.org/anthology/D14-1081.pdf) | [Link](https://github.com/bluemonk482/co-training-for-domain-adaptation) |
### Self-training-based clustering
Year | Title | Venue | Paper | Code 
--- | --- | --- | --- | ---
2014 | A Self-training Approach to Cluster Unlabeled Documents | ACM Transactions on Speech and Language Processing | [Link](https://dl.acm.org/doi/abs/10.1145/2661529) | N/A
2015 | Self-Training Ensemble Clustering for High Dimensional Data | IEEE International Conference on Data Mining | [Link](https://ieeexplore.ieee.org/document/7373341) | [Code](https://github.com/luckiezhou/Self-Training-Ensemble-Clustering)
2016 | Self-Training Ensemble Clustering for High Dimensional Data | IEEE Transactions on Knowledge and Data Engineering | [Link](https://ieeexplore.ieee.org/document/7396918) | N/A
2018 | Self-Training-Based Clustering Ensemble for High Dimensional Data | IEEE Transactions on Neural Networks and Learning Systems | [Link](https://ieeexplore.ieee.org/document/8359461) | [Code](https://github.com/luckiezhou/Self-Training-Ensemble-Clustering)
2019 | Self-Training Ensemble Clustering for High Dimensional Data: A Multi-Objective Optimization Framework | Information Sciences | [Link](https://www.sciencedirect.com/science/article/abs/pii/S0020025519301570) | [Code](https://github.com/luckiezhou/Self-Training-Ensemble-Clustering)
2020 | Self-Training Ensemble Clustering with Consistent Cluster Information | Proceedings of the AAAI Conference on Artificial Intelligence | [Link](https://ojs.aaai.org/index.php/AAAI/article/view/6489/6347) | N/A
2021 | Self-Training Ensemble Clustering with Entropy Regularization | IEEE Transactions on Neural Networks and Learning Systems | [Link](https://ieeexplore.ieee.org/document/9448888) | [Code](https://github.com/mfuca/Self-Training-Ensemble-Clustering-with-Entropy-Regularization)
2022 | Self-Training-Based Ensemble Clustering with Dynamically Consistent Cluster Assignments | Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition | [Link](https://openaccess.thecvf.com/content/ICCV2021/papers/Zhou_Self-Training-Based_Ensemble_Clustering_With_Dynamically_Consistent_Cluster_Assignments_ICCV_2021_paper.pdf) | [Code](https://github.com/luckiezhou/Self-Training-Ensemble-Clustering)
2023 | Self-Training Ensemble Clustering with Improved Diversity | Knowledge-Based Systems | [Link](https://www.sciencedirect.com/science/article/pii/S0950705122003100) | N/A



### Generate semi-supervised models

Year | Title | Venue | Paper | Code 
--- | --- | --- | --- | ---
2014 | Semi-Supervised Learning with Deep Generative Models | Advances in Neural Information Processing Systems | [Link](https://proceedings.neurips.cc/paper/2014/hash/6e8ba87b69b25a1a9b0cf1fe657f29d1-Abstract.html) | N/A
2015 | A Quality-Diversity Algorithm for Semi-Supervised Clustering with Generative Models | IEEE Transactions on Cybernetics | [Link](https://ieeexplore.ieee.org/document/7042659) | N/A
2016 | Improved Techniques for Training GANs | Advances in Neural Information Processing Systems | [Link](https://papers.nips.cc/paper/2016/hash/4a82bceae955be5e8f53c8c155fc32e3-Abstract.html) | [Code](https://github.com/openai/improved-gan)
2017 | A Unified Approach to Semi-Supervised Learning with Generative Adversarial Nets | IEEE Transactions on Pattern Analysis and Machine Intelligence | [Link](https://ieeexplore.ieee.org/document/8016922) | [Code](https://github.com/lzhbrian/SSGAN-Tensorflow)
2018 | Semi-Supervised Deep Generative Models for Improved Scene Understanding | IEEE Transactions on Image Processing | [Link](https://ieeexplore.ieee.org/document/8370201) | N/A
2019 | Semi-Supervised Clustering with DPGMM: The Role of Intrinsic Dimension | Knowledge-Based Systems | [Link](https://www.sciencedirect.com/science/article/pii/S0950705119300442) | N/A
2020 | ClusterGAN++: Learning Discrete Latent Codes for Unsupervised/Semi-Supervised Clustering | IEEE/CVF Conference on Computer Vision and Pattern Recognition | [Link](https://openaccess.thecvf.com/content_CVPR_2020/papers/Choi_ClusterGAN_Learning_Discrete_Latent_Codes_for_Unsupervised_Semi-Supervised_Clustering_CVPR_2020_paper.pdf) | [Code](https://github.com/chrischoy/ClusterGAN)
2021 | Semi-Supervised Feature Assignment using Multi-Assignment GANs | IEEE Journal of Selected Topics in Signal Processing | [Link](https://ieeexplore.ieee.org/document/9404369) | [Code](https://github.com/Bhavya-123/MAGAN)
2022 | Semi-Supervised Clustering via Deep Generative Models with Gumbel-Softmax Trick | Information Sciences | [Link](https://www.sciencedirect.com/science/article/pii/S0020025522006898) | N/A
2023 | Deep Semi-Supervised Learning for Anomaly Detection | Proceedings of the AAAI Conference on Artificial Intelligence | [Link](https://ojs.aaai.org/index.php/AAAI/article/view/18284/18197) | N/A

### S3VMs

Year | Title | Venue | Paper | Code 
--- | --- | --- | --- | ---
2014 | Semisupervised overfitting Control by ℓ1,2-Regularized SVM | IEEE Transactions on Neural Networks and Learning Systems | [Link](https://ieeexplore.ieee.org/document/6784816) | N/A
2015 | Improving Semi-Supervised Learning Performance with Temporal Coherence | AAAI Conference on Artificial Intelligence | [Link](https://ojs.aaai.org/index.php/AAAI/article/view/10816/10667) | N/A
2016 | Efficient Feature Selection for Semi-Supervised SVM with l2 Regularizer | Journal of Information and Computational Science | [Link](http://www.joics.com/uploadfile/2016/0519/20160519023200180.pdf) | N/A
2017 | A Novel Method for Semi-Supervised Classification Based on Adaptive Discrete Artificial Bee Colony Algorithm | Neurocomputing | [Link](https://www.sciencedirect.com/science/article/pii/S0925231217315438) | N/A
2018 | A Semi-Supervised Double Regularized SVM Based on Cliff Delta Score | International Conference on Intelligent Computing and Intelligent Systems | [Link](https://ieeexplore.ieee.org/document/8590859) | N/A
2019 | Semi-Supervised Fraud Detection with High Correlation Feature Selection and Soft Margin SVM | Journal of Information Security and Applications | [Link](https://www.sciencedirect.com/science/article/pii/S2214212618309652) | N/A
2020 | Semi-Supervised Classification of Hyperspectral Images based on Support Vector Machines | Sensors | [Link](https://www.mdpi.com/1424-8220/20/23/7033) | [Code](https://github.com/noelpy/Semi-Supervised-Hyperspectral-Image-Classification-Using-SVM)
2021 | Semi-Supervised Evo-SVM for Interval-Valued Data Classification with Differential Evolution | IEEE Transactions on Fuzzy Systems | [Link](https://ieeexplore.ieee.org/document/9314138) | N/A
2022 | Effect of Non-Normality on Semi-Supervised Support Vector Machines | Neurocomputing | [Link](https://www.sciencedirect.com/science/article/abs/pii/S0925231222010750) | N/A
2023 | Large-Scale Consensus-Based Semi-Supervised Learning with Support Vector Machines | IEEE Transactions on Cybernetics | [Link](https://ieeexplore.ieee.org/document/9571790) | N/A

### Graph-Based Algorithms

Year | Title | Venue | Paper | Code 
--- | --- | --- | --- | ---
2014 | Graph-Based Semi-Supervised Learning with Convolutional Neural Networks | Conference on Computer Vision and Pattern Recognition | [Link](https://ieeexplore.ieee.org/document/6909639) | [Code](https://github.com/dhlee347/pyGSSL)
2015 | Non-Parametric Graph Construction for Semi-Supervised Learning on Manifolds | Advances in Neural Information Processing Systems | [Link](https://proceedings.neurips.cc/paper/2015/hash/2a1f385f7412f0dee5f7aa523f45a8ff-Abstract.html) | [Code](https://github.com/mmazuran/deep-ssl)
2016 | A Framework of Semi-Supervised Learning Based on Deep Generative Models | International Conference on Computer Vision | [Link](https://ieeexplore.ieee.org/document/7780681) | N/A
2017 | Learning Deep Representations with Probabilistic Knowledge Transfer | Ninth International Conference on Machine Learning and Data Mining | [Link](https://link.springer.com/chapter/10.1007/978-3-319-62401-3_2) | [Code](https://github.com/bharel/SSNMTL)
2018 | Combination of Edge and Node Information for Semi-Supervised Learning with Graph Convolutional Networks | IEEE Access | [Link](https://ieeexplore.ieee.org/document/8343583) | [Code](https://github.com/kimiyoung/planetoid)
2019 | Graphology: A Fast and Scalable Framework for Graph-Regularized Deep Learning-Based Drug Discovery | Journal of Chemical Information and Modeling | [Link](https://pubs.acs.org/doi/10.1021/acs.jcim.9b00673) | N/A
2020 | Learning with Graphs: A Survey | Foundations and Trends in Machine Learning | [Link](https://www.nowpublishers.com/article/Details/MAL-084) | N/A
2021 | Semi-Supervised Learning with Graph Convolutional Networks: Methods, Analysis, and Applications | Journal of Signal Processing Systems | [Link](https://link.springer.com/article/10.1007/s11265-020-01767-5) | N/A
2022 | Graph Convolutional Networks with FishNet Graph Construction for Semi-Supervised Classification | IEEE Transactions on Neural Networks and Learning Systems | [Link](https://ieeexplore.ieee.org/document/9576872) | N/A
2023 | Graph Regularized Semi-Supervised Learning with Cross-Modal Representation | Information Sciences | [Link](https://www.sciencedirect.com/science/article/pii/S0020025522002593) | N/A



## Benchmark Datasets

Since semi-supervised clustering is primarily applied to graph-structured data, there are no ‘non-graph datasets’ available for this purpose. Typically, semi-supervised learning algorithms are only used in graph data because only graph data can naturally define neighborhood relationships.

#### Quick Start

- Step1: Download all datasets from \[[Google Drive](https://drive.google.com/drive/folders/1thSxtAexbvOyjx-bJre8D4OyFKsBe1bK?usp=sharing) | [Nutstore](https://www.jianguoyun.com/p/DfzK1pwQwdaSChjI2aME)]. Optionally, download some of them from URLs in the tables (Google Drive)
- Step2: Unzip them to **./dataset/**
- Step3: Change the type and the name of the dataset in **main.py**
- Step4: Run the **main.py**

- **utils.py**
  1. **load_data**: load graph datasets 
  2. **preprocess_data**:  performs additional preprocessing on the data
  3. **generate_labels**: generates pseudolabels or predicted labels
  4. **train_model**:  trains the semi-supervised clustering model on the data
  5. **predict_clusters**: predicts the cluster assignments for the unlabeled data
  6. **evaluate_model**: evaluates the performance of the model on test data
  7. **plot_clusters**: visualizes the cluster assignments and/or centroids in a scatter plot, heatmap, or other graphical representation
  8. **save_model and load_model**:   computes various clustering performance metrics
  9. **compute_cluster_metrics**: evaluate the performance of clustering
  10. **normalize_data**: normalizes the data for better clustering performance


#### Datasets Details

About the introduction of each dataset, please check [here](./dataset/README.md)
| Dataset | # Samples | # Dimension | # Edges | # Classes | URL |
|---------|-----------|-------------|---------|-------------|-----|
| MNIST   | 70,000    | 784         | N/A     | 10         | [Link](http://yann.lecun.com/exdb/mnist/) |
| Fashion-MNIST | 70,000 | 784 | N/A | 10 | [Link](https://github.com/zalandoresearch/fashion-mnist) |
| CIFAR-10 | 60,000    | 3 * 32 * 32 | N/A | 10 | [Link](https://www.cs.toronto.edu/~kriz/cifar.html) |
| CIFAR-100 | 60,000   | 3 * 32 * 32 | N/A | 100 | [Link](https://www.cs.toronto.edu/~kriz/cifar.html) |
| SVHN    | 600,000   | 32 * 32 * 3 | N/A | 10         | [Link](http://ufldl.stanford.edu/housenumbers/) |
| Reuters-21578 | 21,578 | 2,000 | N/A | 90 | [Link](https://archive.ics.uci.edu/ml/datasets/Reuters-21578+Text+Categorization+Collection) |
| 20 Newsgroups | 20,000  | N/A         | N/A     | 20         | [Link](http://qwone.com/~jason/20Newsgroups/) |
| Olivetti faces | 400      | 64 * 64 | N/A     | 40         | [Link](https://scikit-learn.org/stable/datasets/) |
Citeseer | 3327 | 3703 | 4536 | 6 | [Link](https://linqs-data.soe.ucsc.edu/public/lbc/citeseer.tgz)
Cora | 2708 | 1433 | 5429 | 7 | [Link](https://linqs-data.soe.ucsc.edu/public/lbc/cora.tgz)
DBLP | 1775 | 334 | 9005 | 4 | [Link](https://linqs-data.soe.ucsc.edu/public/dblp.tgz)
Pubmed | 19717 | 500 | 44327 | 3 | [Link](https://linqs-data.soe.ucsc.edu/public/lbc/pubmed.tgz)
NELL | 2389 | 542 | 2763 | 9 | [Link](https://www.dropbox.com/s/wi7xat1rrr8hq4j/ReadMe.txt?dl=0)
Wisconsin Breast Cancer | 569 | 30 | 1980 | 2 | [Link](https://www.cs.wisc.edu/~street/729/Project/WBCD.tgz)
USPS | 9298 | 256 | 180714 | 10 | [Link](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass.html#usps)
WebKB | 877 | 1703 | 1603 | 5 | [Link](https://linqs-data.soe.ucsc.edu/public/wcb.tgz)
BlogCatalog | 10312 | 3703 | 333983 | 39 | [Link](https://linqs-data.soe.ucsc.edu/public/lbc/BlogCatalog-dataset.rar)
Flickr | 89250 | 500 | 899756 | 195 | [Link](http://webdatacommons.org/hyperlinkgraph/2014-04/download.html)

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
