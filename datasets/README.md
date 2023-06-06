## Benchmark Datasets

#### Datasets Details

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



#### Dataset Introduction

##### MNIST

A collection of handwritten digits, consisting of 70,000 28x28 grayscale images, divided into 10 different classes, with each class representing a digit.

##### Fashion-MNIST

A dataset of fashion items’ images, including 70,000 28x28 grayscale images, which are divided into 10 different classes, with each class representing a specific fashion item.

##### CIFAR-10

A dataset of images for classification, consisting of 60,000 32x32 RGB images divided into 10 different classes, with each class representing a set of common images.

##### CIFAR-100

A dataset of images for classification, consisting of 60,000 32x32 RGB images divided into 100 different classes, with each class representing a more fine-grained set of images.

##### SVHN

A collection of handwritten digits, consisting of 600,000 32x32 RGB images extracted from Google Street View and other sources, divided into 10 different classes, with each
class representing a digit.

##### Reuters-21578

A dataset of multi-labeled news text provided by Reuters. The dataset includes 21,578 documents, each of which is labeled with one or more topics.

##### 20 Newsgroups

A text classification dataset consisting of 20,000 news posts from 20 different news groups, with each group representing a different topic.

##### Olivetti faces

A dataset of facial images, consisting of 400 64x64 black and white images, each of which is a facial image of the same person in different directions and expressions.

##### Citeseer

The Citeseer dataset is an academic paper network dataset that contains 3,327 nodes and 4,536 edges. Each node represents an academic paper, and edges represent citation
relationships between these papers. Each node has a label that represents the field of study to which the paper belongs, and there are six different fields.

##### Cora

The Cora dataset is also an academic paper network dataset that contains 2,708 nodes and 5,429 edges. Each node represents an academic paper, and edges represent citation
relationships between these papers. Each node has a label that represents the field of study to which the paper belongs, and there are seven different fields.

##### DBLP

This dataset is also an academic paper network dataset that contains 1,775 nodes and 9,005 edges. Each node represents an academic paper, and edges represent citation
relationships between these papers. Each node has four predefined class labels: male author, female author, machine learning, and data mining.

##### Pubmed

The Pubmed dataset is a medical literature network dataset that contains 19,717 nodes and 44,327 edges. Each node represents a medical paper, and edges represent the
relationship between these papers. Each node has three predefined class labels, indicating the research topic of the paper.

##### NELL

This dataset is a subset of the NELL knowledge base and contains 2,389 nodes and 2,763 edges. Each node represents an entity, and edges represent relationships between
these entities.

##### Wisconsin Breast Cancer 

The Wisconsin Breast Cancer dataset is a medical dataset that contains 569 nodes and 1,980 edges. Each node represents a patient, with 30 features. This dataset was obtained
by evaluating breast cell biopsies.

##### USPS

The USPS dataset is a handwritten digit image dataset that contains 9,298 nodes and 180,714 edges. Each node represents a handwritten digit image. Each node has a predefined
label, indicating which digit (0-9) the image represents.

##### WebKB

The WebKB dataset is a web page classification dataset that contains 877 nodes and 1,603 edges. Each node represents a web page, and edges represent the link relationships between
these web pages. Each node has a label, indicating the category to which the web page belongs, with five different categories.

##### BlogCatalog 

The BlogCatalog dataset is a social network dataset that contains 10,312 nodes and 333,983 edges. Each node represents a blog, and edges represent the common tags between these
blogs. Each node has 39 labels, describing the content of the blog.

##### Flickr

The Flickr dataset is an image tag network dataset that contains 89,250 nodes and 899,756 edges. Each node represents an image, and edges represent the similarity relationships
between these images. Each node has a label, indicating the topic of the image, with a total of 195 topics.









If you find this repository useful to your research or work, it is really appreciate to star this repository.​ :heart:
