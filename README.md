# Machine Learning in Nine Days

## Installations
For this mini-course you'll only be needing to install [**Anaconda**](https://www.anaconda.com/distribution/#download-section).
Please install the Python 3.x version. 
You do not need to install PyCharm (or any IDE, for that matter) since all exercises will be done in Jupyter notebooks. I recommend you get yourself acquainted with the keyboard shortcuts, it will make your life much easier. 

*Important: when the installation asks if you want to add Anaconda to your PATH, click **yes**. This would make your life easier later. If you already have another version of python installed on your computer, I recommend that you uninstall it first and only then install the Anaconda version, to prevent weird things from happening due to collisions.*

## Syllabus

|  Day  | Subjects | Materials | Exercises |
| :---: | --- | --- | --- |
| 0 | Python | 1. Stanfords CS231N [tutorial](http://cs231n.github.io/python-numpy-tutorial/) <br> 2. [Python for DS](Day%200/1%20Python/Python_Basic_Concepts.ipynb) |  |
| 0 | Numpy | 1. Stanfords CS231N [tutorial](http://cs231n.github.io/python-numpy-tutorial/) (Pay extra attention to indexing, slicing, and **broadcasting**)  | 1. [Numpy-100](Day%200/2%20Numpy/numpy-100) <br> Recommended exercises: <br> &nbsp;&nbsp;&nbsp;a. From exercises 1-31, do *everything but* 2, 4, 5, 23, 27, 28, 31. <br> &nbsp;&nbsp;&nbsp;b. From 32-51, do *only* 35, 37, 38, 39, 40, 45, 46, 50. <br> &nbsp;&nbsp;&nbsp;c. From 52-100, do *only* 52, 53, 54, 55, 58, 59, 61, 64, 65, 70, 71, 72, 74, 75, 83, 89.|
| 0 | Pandas | 1. [10 minutes to pandas](https://pandas.pydata.org/pandas-docs/stable/getting_started/10min.html) | 1. [Pandas exercises](Day%200/3%20Pandas/pandas_exercises). Follow the instructions in the README file. No need to do everything, stop whenever you feel you've done enough. <br> 2. [Titanic_visualization](Day%200/3%20Pandas/Titanic_visualization): a nice exercise you are free to play with. Follow the instructions in the README file. |
| 0 | Linear Algebra | 1. Stanford's CS229 [review](http://cs229.stanford.edu/section/cs229-linalg.pdf). | |
| 0 | Probability and Statistics | 1. Stanford's CS229 [review](http://cs229.stanford.edu/section/cs229-prob.pdf) <br> 2. (Optional) read online on the normal distribution and multivariate normal distributions. [This](http://cs229.stanford.edu/section/gaussians.pdf) and [This](http://cs229.stanford.edu/section/more_on_gaussians.pdf) might be useful, but they might be a bit too "mathematical" for our purposes. | |
| 1 | Intro: Supervised VS Unsupervised Learning | 1. [Presentation](Day%201/1%20Intro%20-%20Supervised%20vs%20Unsupervised/Supervised%20vs%20Unsupervised.pptx) | 1. [Introduction](Day%201/1%20Intro%20-%20Supervised%20vs%20Unsupervised/Exercises) |
| 1 | Unsupervised: Clustering using k-means and GMM | 1. [Presentation](Day%201/2%20Clustering/Clustering.pptx) | 1. [Clustering exercise](Day%201/2%20Clustering/exercise) |
| 1 | Unsupervised: Dimensionality reduction using PCA and SVD | 1. [Presentation on SVD](Day%201/3%20Dimensionality%20reduction/SVD.pptx) <br> 2. [Presentation on PCA](Day%201/3%20Dimensionality%20reduction/PCA.pptx) <br> 3. [Intuition to PCA](https://stats.stackexchange.com/questions/2691/making-sense-of-principal-component-analysis-eigenvectors-eigenvalues/140579#140579) <br> 4. [PCA using SVD](https://stats.stackexchange.com/questions/134282/relationship-between-svd-and-pca-how-to-use-svd-to-perform-pca) <br> 4. [Notes](https://webcourse.cs.technion.ac.il/234125/Winter2018-2019/ho/WCFiles/Tutorial-Eigs_SVD.pdf) from Technion's Numerical Analysis course | 1. [Exercise](Day%201/3%20Dimensionality%20reduction/exercise): representing faces efficiently |
| 1 | Supervised: Classification using KNN and linear classifiers | 1. [Presentation](Day%201/4%20Classification/Linear%20classifiers.pptx) on linear classifiers and decision trees <br> 2. CS231n [lecture 2 slides](http://cs231n.stanford.edu/slides/2017/cs231n_2017_lecture2.pdf) on KNN and choosing the right k using cross validation | 1. [Andrew Ng's exercises](Day%201/4%20Classification/Andrew's%20Exercises) <br> 2. [KNN exercise](Day%201/4%20Classification/KNN%20exercise) **TODO** <br> 3. [Linear classifier vs KNN exercise](Day%201/4%20Classification/linear%20classifier%20vs%20knn%20exercise) **TODO** |
| 1 | Supervised: Linear and Logistic regressions | 1. Linear regression [presentation](Day%201/5%20Regression/1%20Linear%20Regression.pptx) <br> 2. Logistic regression [presentation](Day%201/5%20Regression/2%20Logistic%20Regression.pptx) | 1. [Andrew Ng's exercises](Day%201/5%20Regression/Andrew's%20exercises) <br> 2. [Linear regression using Least Squares](Day%201/5%20Regression/exercise%20-%20Linear%20regression%20using%20LS) **TODO** <br> 3. [Logistic regression on MNIST](Day%201/5%20Regression/exercise%20-%20Logistic%20regression%20on%20MNIST) **TODO** |
| 1 | Intro to Optimization: Gradient Descent | 1. [Presentation](Day%201/6%20Intro%20to%20Optimization/Intro%20to%20Optimization.pptx) | |
| 2 | Computation graphs and back propagation | CS231n [lecture 4 slides](http://cs231n.stanford.edu/slides/2017/cs231n_2017_lecture4.pdf) | |
| 2 | Logistic regression using a single neuron | 1. [  Presentation](Day%202/01%20Logistic%20regression%20with%20one%20neuron.pptx) | 1. [Exercise](Day%202/02%20exercise%20-%20Logistic%20Regression%20with%20NN) |
| 2 | Deep NN | 1. [Presentation](Day%202/03%20Deep%20Neural%20Network.pptx)  | 1. [Implementing a NN in Numpy](Day%202/04%20exercise%20-%20Building%20your%20Deep%20Neural%20Network%20-%20Step%20by%20Step) <br> 2. [Using a NN for image classification](Day%202/05%20exercise%20-%20Deep%20Neural%20Network%20Application_%20Image%20Classification) <br> 3. [Planar data classification exercise](Day%202/06%20exercise%20-%20Planar%20data%20classification%20with%20one%20hidden%20layer) |
| 3 | Activation functions | 1. CS231n [lecture 6 slides](http://cs231n.stanford.edu/slides/2017/cs231n_2017_lecture6.pdf) (13-33) | |
| 3 | Data preprocessing | 1. CS231n [lecture 6 slides](http://cs231n.stanford.edu/slides/2017/cs231n_2017_lecture6.pdf) (34-39) | |
| 3 | Weight initialization | 1. CS231n [lecture 6 slides](http://cs231n.stanford.edu/slides/2017/cs231n_2017_lecture6.pdf) (40-52)| |
| 3 | Batch normalization | 1. CS231n [lecture 6 slides](http://cs231n.stanford.edu/slides/2017/cs231n_2017_lecture6.pdf) (53-60) <br> 2. [Presentation](Day%203/Batch%20normalization.pptx) | |
| 3 | The learning process | 1. CS231n [lecture 6 slides](http://cs231n.stanford.edu/slides/2017/cs231n_2017_lecture6.pdf) (61-74) | |
| 3 | Hyperparameter tuning | 1. CS231n [lecture 6 slides](http://cs231n.stanford.edu/slides/2017/cs231n_2017_lecture6.pdf) (75-88) <br> 2. [Presentation](Day%203/Hyperparameters%20tuning.pptx) | |
| 3 | Advanced optimization: Momentum, RMSprop, AdaGrad, Adam | 1. CS231n [lecture 7 slides](http://cs231n.stanford.edu/slides/2017/cs231n_2017_lecture7.pdf) (14-42) <br> 2. [Presentation](Day%203/Optimization%20Algorithms.pptx) | |
| 3 | Regularization: loss function terms, dropout, data augmentation, stochastic depth | 1. CS231n [lecture 7 slides](http://cs231n.stanford.edu/slides/2017/cs231n_2017_lecture7.pdf) (58-85) <br> 2. [Presentation](Day%203/Setting%20up%20ML%20applications.pptx) | |
| 3 | Softmax | 1. [Presentation](Day%203/Multi-class%20classification%20using%20Softmax.pptx) | |
| 3 | Transfer Learning | 1. CS231n [lecture 7 slides](http://cs231n.stanford.edu/slides/2017/cs231n_2017_lecture7.pdf) (86-93,97) | |
| 3 | Tensorflow |  | 1. [Tensorflow basics](Day%203/Tensorflow%20exercise) <br> 2. [Implementing a NN in Tensorflow](Day%203/Tensorflow%20exercise) |
| 3 | CNNs | 1. CS231n [lecture 5 slides](http://cs231n.stanford.edu/slides/2017/cs231n_2017_lecture5.pdf) <br> 2. [Presentation](Day%203/10%20CNNs/CNN%20Overview.pptx) | 1. [Implementing a CNN in Tensorflow](Day%203/10%20CNNs/exercise%20-%20Implementing%20a%20CNN) <br> 2. [Hand sign recognition](Day%203/10%20CNNs/exercise%20-%20Hand%20sign%20recognition) |
| 4 | Keras | | 1. [Happy house 1](Day%204/exercise%20-%20Keras%20Happy%20House) |
| 4 | Classic CNN Architectures: LeNet, AlexNet, VGG, GoogLeNet, ResNet | 1. CS231n [lecture 9 slides](http://cs231n.stanford.edu/slides/2017/cs231n_2017_lecture9.pdf) <br> 2. [Presentation](Day%204/Classic%20Architectures.pptx) | 1. [ResNet exercise](Day%204/exercise%20-%20ResNet) |
| 4 | Keras 2 | | 1. [Happy house 2](Day%204/exercise%20-%20Happy%20House%202). Before the exercise, go over [this presentation](Day%204/Before%20the%20face%20recognition%20exercise.pptx) |
| 4 | Visualizations of CNNs | 1. CS231n [lecture 12 slides](http://cs231n.stanford.edu/slides/2017/cs231n_2017_lecture12.pdf) | 1. [Visualizations of hand sign recognition network]() **TODO** |
| 4 | Adversarial Examples | 1. CS231n [lecture 16 slides](http://cs231n.stanford.edu/slides/2017/cs231n_2017_lecture16.pdf)| |

## Additional Material
1. Stanford's CS231n (CNNs for Visual Recognition) is ***highly recommended***. The [course website](http://cs231n.stanford.edu/) contains lots of useful material, and recorded lectures can be found on [YouTube](https://www.youtube.com/playlist?list=PL3FW7Lu3i5JvHM8ljYj-zLfQRF3EO8sYv).
1. Stanford's CS224n (NLP with Deep Learning) is also ***highly recommended***. Useful material can be found on the [course website](http://web.stanford.edu/class/cs224n/), and the lectures can be found on [YouTube](https://www.youtube.com/playlist?list=PL3FW7Lu3i5Jsnh1rnUwq_TcylNr7EkRe6).
1. Andrew Ng's [Machine learning course](https://www.coursera.org/learn/machine-learning) on Coursera is good. Also on Coursera, you can his series of [5 mini-courses in deep learning](https://www.coursera.org/specializations/deep-learning), which is also very good.
1. [This website](http://homepages.inf.ed.ac.uk/rbf/CVonline/Imagedbase.htm) that contains many datasets that are in standard use in Machine Learning research (mostly for computer vision).
1. [Google's Into to ML Crash Course](https://developers.google.com/machine-learning/crash-course/ml-intro)
1. [How to Apply ML to Business Problems](https://emerj.com/ai-executive-guides/how-to-apply-machine-learning-to-business-problems/)
1. [How to define your ML problem](https://machinelearningmastery.com/how-to-define-your-machine-learning-problem/)
1. [Kaggle - Decision Trees and Random Forests](https://campus.datacamp.com/courses/kaggle-python-tutorial-on-machine-learning)
1. [Kaggle - Unsupervised Learning in Python](https://campus.datacamp.com/courses/unsupervised-learning-in-python)
1. [The 5 Clustering Algorithms Data Scientists Need To Know](https://towardsdatascience.com/the-5-clustering-algorithms-data-scientists-need-to-know-a36d136ef68) 