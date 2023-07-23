# SKLearn ML Modelserver

<!-- TOC -->

- [SKLearn ML Modelserver](#sklearn-ml-modelserver)
  - [Quickstart](#quickstart)
    - [SGD Classifier](#sgd-classifier)
    - [kNN Classifier](#knn-classifier)
    - [Dataset](#dataset)
  - [Build your own Classifier](#build-your-own-classifier)
    - [Step 0 - Dataset Preprocessing](#step-0---dataset-preprocessing)
    - [Step 1 - SGD Model Preparation](#step-1---sgd-model-preparation)
    - [Step 1 - SGD Model Prediction Pipeline](#step-1---sgd-model-prediction-pipeline)
    - [Step 2 - And so on...](#step-2---and-so-on)
    - [Step 3 - Deployment](#step-3---deployment)
      - [Potential Error](#potential-error)

<!-- /TOC -->

## Quickstart

> __Update__: I had to remove the model and dataset as they were to big for Github. So this Quickstart does not work unless you download the Flower102 dataset, label it and run the scripts as explained below to generate the models needed for this container ¯\\_ (ツ)_/¯


```bash
cd ./STEP3_Webfrontend_Deployment
docker build -t modelserver .
docker run -p 5000:5000 modelserver:latest
```

Visit `localhost:5000` to access the web interface:


![SKLearn ML Modelserver](https://github.com/mpolinowski/i-know-flowers/blob/master/STEP3_Webfrontend_Deployment/static/images/README_01.webp)


Select the model you want to use and upload a close-up photo of a flower contained in the map classes the model was trained on.


### SGD Classifier

The Stochastic Gradient Descent classifier was trained to an accuracy of `29.4%` over all 45 class labels:


![SKLearn ML Modelserver](https://github.com/mpolinowski/i-know-flowers/blob/master/STEP3_Webfrontend_Deployment/static/images/README_02.webp)


### kNN Classifier

The k-Nearest-Neighbors classifier was trained to an accuracy of `36.7%` over all 45 class labels. As seen from the confusion matrix the confusion is much more localized to a handful of classes and could probably be remedied by cleaning up / extending the dataset for those classes:


![SKLearn ML Modelserver](https://github.com/mpolinowski/i-know-flowers/blob/master/STEP3_Webfrontend_Deployment/static/images/README_03.webp)


### Dataset

Under the __About__ tab you see an overview of all included classes and samples of the images used to train the classifiers - all images used were taken from the [Flowers102 Dataset](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html):


![SKLearn ML Modelserver](https://github.com/mpolinowski/i-know-flowers/blob/master/STEP3_Webfrontend_Deployment/static/images/README_04.webp)



## Build your own Classifier

### Step 0 - Dataset Preprocessing

Collect images and place them into sub folders in `/STEP0_Dataset_Preprocessing/labelled_data` based on their classes. Go though the [Python notebook](/STEP0_Dataset_Preprocessing) to prepare the dataset pickle.

### Step 1 - SGD Model Preparation

The [next step](/STEP1_SGD_01_Model_Preparation) is to prepare your data by performing a train/test-split and extract a feature vector from each image using the Hog transformer. Prepare the SGD Classifier (can be replaced with any classifier from SKLearn that you want to use). Fit it to the feature vectors from your dataset.

Once you have a working model perform a grid search to find the optimal hyperparameter. Re-fit the model using those parameter and export the trained model ready for deployment.

### Step 1 - SGD Model Prediction Pipeline

In the [next step](/STEP1_SGD_02_Model_Prediction_Pipeline) you need to develop the prediction pipeline that you will use for the Flask App Prediction API later on.


### Step 2 - And so on...

Add as many classifiers as you need to find one that performs the best - I added the k-Nearest-Neighbors classifier that outperformed the SGD model. Depending on your dataset you might experience the opposite or need another classifier all-together.


### Step 3 - Deployment

With those models prepared we need a [Flask Backend](/STEP3_Webfrontend_Deployment) to serve them as a prediction API with a simple HTML user interface. Add routes for your own classifiers - if you added them above. And jump back up to [Quickstart](#quickstart) to build and deploy your service container!


#### Potential Error

If you see error messages like the following when trying to start your container:

> `/usr/local/lib/python3.11/site-packages/sklearn/base.py:347: InconsistentVersionWarning: Trying to unpickle estimator SGDClassifier from version 1.2.2 when using version 1.3.0. This might lead to breaking code or invalid results. Use at your own risk. For more info please refer to:
https://scikit-learn.org/stable/model_persistence.html#security-maintainability-limitations`

> `AttributeError: Can't get attribute 'ManhattanDistance' on <module 'sklearn.metrics._dist_metrics' from '/usr/local/lib/python3.11/site-packages/sklearn/metrics/_dist_metrics.cpython-311-x86_64-linux-gnu.so'>`


Make sure that the version of Scikit-Learn on your system is identical to the version installed inside the container! Edit the version used in `requirements.txt` accordingly.