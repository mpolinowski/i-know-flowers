from flask import Flask, render_template, request, redirect, url_for
import os
import pickle
import numpy as np
import pandas as pd
import scipy as sc
from sklearn.base import BaseEstimator, TransformerMixin
from skimage import io, color, transform, feature

PORT = 3000

BASE_PATH = os.getcwd()
UPLOAD_PATH = os.path.join(BASE_PATH,'static/uploads/')
MODELS_PATH = os.path.join(BASE_PATH,'static/models/')

SGD_MODEL = os.path.join(MODELS_PATH,'sgd_model_deployment.pkl')
SGD_SCALER = os.path.join(MODELS_PATH,'sgd_model_deployment_scaler.pkl')

model_sgd = pickle.load(open(SGD_MODEL,'rb'))
normalizer_sgd = pickle.load(open(SGD_SCALER,'rb'))

KNN_MODEL = os.path.join(MODELS_PATH,'knn_model_deployment.pkl')
KNN_SCALER = os.path.join(MODELS_PATH,'knn_model_deployment_scaler.pkl')

model_knn = pickle.load(open(KNN_MODEL,'rb'))
normalizer_knn = pickle.load(open(KNN_SCALER,'rb'))

app = Flask(__name__)



@app.errorhandler(404)
def error404(error):
    error_title = "404 Not Found"
    error_message = "The server cannot find the requested resource. In the browser, this means the URL is not recognized. In an API, this can also mean that the endpoint is valid but the resource itself does not exist."
    return render_template("error.html", title=error_title, message=error_message)

@app.errorhandler(405)
def error405(error):
    error_title = "405 Method Not Allowed"
    error_message = 'The request method is known by the server but is not supported by the target resource.'
    return render_template("error.html", title=error_title, message=error_message)

@app.errorhandler(500)
def error500(error):
    error_title = "500 Internal Server Error"
    error_message='This error response means that the server, while working as a gateway to get a response needed to handle the request, got an invalid response.'
    return render_template("error.html", title=error_title, message=error_message)


@app.route('/')
def index():
    return render_template('eval.html')


@app.route('/about/')
def about():
    return render_template('about.html')



@app.route('/sgd/', methods=['GET','POST'])
def sgd():
    # upload file
    if request.method == "POST":
        upload_file = request.files['image']
        extension = upload_file.filename.split('.')[-1]
        if extension.lower() in ['png', 'jpg', 'jpeg']:
            file_path = os.path.join(UPLOAD_PATH,upload_file.filename)
            upload_file.save(file_path)
        else:
            print('ERROR :: File extension not allowed,', extension)
            return render_template('upload_sgd.html', fileupload=False, extexception=True, extension=extension)
        print('INFO :: File uploaded', upload_file.filename)
        # run prediction
        results = prediction_pipeline_sgd(file_path, normalizer_sgd, model_sgd)
        img_width = calc_width(file_path)
        print('INFO :: Prediction results', results)
        return render_template(
            'upload_sgd.html',
            fileupload=True,
            extexception=False,
            image='/static/uploads/'+upload_file.filename,
            data=list(list(results)[0].items()),
            width=img_width
        )
    else:
        return render_template('upload_sgd.html', fileupload=False, extexception=False)



@app.route('/knn/', methods=['GET','POST'])
def knn():
    # upload file
    if request.method == "POST":
        upload_file = request.files['image']
        extension = upload_file.filename.split('.')[-1]
        if extension.lower() in ['png', 'jpg', 'jpeg']:
            file_path = os.path.join(UPLOAD_PATH,upload_file.filename)
            upload_file.save(file_path)
        else:
            print('ERROR :: File extension not allowed,', extension)
            return render_template('upload_knn.html', fileupload=False, extexception=True, extension=extension)
        print('INFO :: File uploaded', upload_file.filename)
        # run prediction
        results = prediction_pipeline_knn(file_path, normalizer_knn, model_knn)
        img_width = calc_width(file_path)
        print('INFO :: Prediction results', results)
        return render_template(
            'upload_knn.html',
            fileupload=True,
            extexception=False,
            image='/static/uploads/'+upload_file.filename,
            data=list(list(results)[0].items()),
            width=img_width
        )
    else:
        return render_template('upload_knn.html', fileupload=False, extexception=False)




def prediction_pipeline_sgd(img_path, normalizer, model):
    img = io.imread(img_path, as_gray=True)
    img_resized = (transform.resize(img, (80, 80)) * 255).astype(np.uint8)
    
    feature_vector = feature.hog(
            img_resized,
            orientations=10,
            pixels_per_cell=(7, 7),
            cells_per_block=(3, 3)
    ).reshape(1, -1)
    
    feature_vector_scaled = normalizer.transform(feature_vector)
    model.predict(feature_vector_scaled)
    
    decision_values = model.decision_function(feature_vector_scaled)
    z_scores = sc.stats.zscore(decision_values.flatten())
    probabilities = (sc.special.softmax(z_scores) * 100).round(2)
    
    labels = model.classes_

    probabilities_df = pd.DataFrame(probabilities, columns=['probability [%]'], index=labels)
    top5predictions = probabilities_df.sort_values(
        by='probability [%]', ascending=False
    )[:5].to_dict().values()
    
    return top5predictions




def prediction_pipeline_knn(img_path, normalizer, model):
    img = io.imread(img_path, as_gray=True)
    img_resized = (transform.resize(img, (80, 80)) * 255).astype(np.uint8)
    
    feature_vector = feature.hog(
            img_resized,
            orientations=10,
            pixels_per_cell=(7, 7),
            cells_per_block=(3, 3)
    ).reshape(1, -1)
    
    feature_vector_scaled = normalizer.transform(feature_vector)
    model.predict(feature_vector_scaled)
    
    decision_values = model.predict_proba(feature_vector_scaled)
    z_scores = sc.stats.zscore(decision_values.flatten())
    probabilities = (sc.special.softmax(z_scores) * 100).round(2)
    
    labels = model.classes_

    probabilities_df = pd.DataFrame(probabilities, columns=['probability [%]'], index=labels)
    top5predictions = probabilities_df.sort_values(
        by='probability [%]', ascending=False
    )[:5].to_dict().values()
    
    return top5predictions


def calc_width(path):
    img = io.imread(path)
    height,width,_ = img.shape
    aspect_ratio = width/height
    
    max_height = 335
    max_width = 360
    optimal_width =  max_height * aspect_ratio

    if optimal_width <= max_width:
        return optimal_width
    else:
        return max_width


if __name__ == 'main':
    app.run(host="localhost", port=PORT, debug=True)