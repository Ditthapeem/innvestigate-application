from __future__ import absolute_import
from flask import (
    Flask, Blueprint, render_template, request, jsonify, send_file
)
import flask_uploads
from flask_cors import CORS, cross_origin
import shutil
import time

import os
import json
import time
import logging
from math import sqrt
from scipy.spatial.distance import cosine

import innvestigate
import innvestigate.utils

import tensorflow
from tensorflow.keras import preprocessing, applications
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.utils import get_file
from tensorflow.keras.layers import GlobalMaxPooling2D

# from tensorflow.convolutional import _Conv

import numpy as np
import matplotlib as mplib
import matplotlib.pyplot as plt
import cv2

from .classes.Response import Response
from .classes.ClassificationMapping import ClassificationMapping
from .classes.Visualization import Visualization
from .classes.Suggestion import Suggestion

tensorflow.compat.v1.disable_eager_execution()

mplib.use('agg')

bp = Blueprint('main', __name__)
CORS(bp)
CORS(bp, origins="*")

images_UploadSet = flask_uploads.UploadSet('images', flask_uploads.IMAGES)
models_UploadSet = flask_uploads.UploadSet('models', 'h5')
class_indexes_UploadSet = flask_uploads.UploadSet('indexes', 'json')


# from iNNvestigate + gradcam and guided gradcam
methods = {
    # "input",  # unnecessary in our tool
    # "random",  # unnecessary in our tool
    "gradient",
    # "gradient.baseline",  # mainly for debugging purposes
    "input_t_gradient",
    "deconvnet",
    "guided_backprop",
    "gradcam",
    "guided_gradcam",
    "integrated_gradients",
    "smoothgrad",
    "lrp",
    "lrp.z",
    "lrp.z_IB",
    "lrp.epsilon",
    "lrp.epsilon_IB",
    "lrp.w_square",
    "lrp.flat",
    "lrp.alpha_beta",
    "lrp.alpha_2_beta_1",
    "lrp.alpha_2_beta_1_IB",
    "lrp.alpha_1_beta_0",
    "lrp.alpha_1_beta_0_IB",
    "lrp.z_plus",
    "lrp.z_plus_fast",
    "lrp.sequential_preset_a",
    "lrp.sequential_preset_b",
    "lrp.sequential_preset_a_flat",
    "lrp.sequential_preset_b_flat",
    "deep_taylor",
    "deep_taylor.bounded",
    # "deep_lift.wrapper",  # DeepLIFT no more supported by the tool
    "pattern.net",
    "pattern.attribution",
}

neuron_selection_modes = {
    "max_activations",
    "index",
}

valid_n_max_activations = {
    "1",
    "4",
    "9",
    "16",
}

# from iNNvestigate
lrp_rules = {
    "Z",
    "ZIgnoreBias",
    
    "Epsilon",
    "EpsilonIgnoreBias",
    
    "WSquare",
    "Flat",
    
    "AlphaBeta",
    "AlphaBetaIgnoreBias",
    
    "Alpha2Beta1",
    "Alpha2Beta1IgnoreBias",
    "Alpha1Beta0",
    "Alpha1Beta0IgnoreBias",
    
    "ZPlus",
    "ZPlusFast",
    "Bounded",
}

# from iNNvestigate
valid_gradient_postprocesses = {
    "None",
    "abs",
    "square",
}

# set app logger
logging.basicConfig(filename='app.log', filemode='w', format='[%(asctime)s] %(levelname)s in %(module)s: %(message)s')
app = Flask(__name__)
CORS(app)


# static paths
STATIC_PATH = "innvestigate-gui/static/"
FRONTEND_INPUT_PATH = "/frontend/public/images/input"
FRONTEND_OUTPUT_PATH = "/frontend/public/images/output"

INPUT_PATH = STATIC_PATH + "input/"
INPUT_IMAGES_PATH = INPUT_PATH + "images/"
INPUT_INDEXES_PATH = INPUT_PATH + "indexes/"
INPUT_MODELS_PATH = INPUT_PATH + "models/"

OUTPUT_PATH = STATIC_PATH + "output/"
OUTPUT_IMAGES_PATH = OUTPUT_PATH + "images/"


def grad_cam(model, image, cls, conv_layer_index, model_input_shape):
    """
    GradCAM method for visualizing input saliency based on code from https://github.com/eclique/keras-gradcam.

    :param model: model on which to apply GradCAM
    :param image: input image
    :param cls: code of class for which to visualize feature of the input image using GradCAM
    :param conv_layer_index: only convolutional layers can be used for visualization
    :param model_input_shape: for properly resizing generated heatmap
    :return: GradCAM map
    """
    
    y_c = model.output[0, cls]
    conv_output = model.get_layer(index=conv_layer_index).output
    
    grads = K.gradients(y_c, conv_output)[0]
    
    gradient_function = K.function([model.input], [conv_output, grads])
    output, grads_val = gradient_function([image])
    output, grads_val = output[0, :], grads_val[0, :, :, :]

    weights = np.mean(grads_val, axis=(0, 1))
    cam = np.dot(output, weights)
    
    # Process CAM
    cam = cv2.resize(cam, model_input_shape, cv2.INTER_LINEAR)
    cam = np.maximum(cam, 0)
    cam = cam / cam.max()  # cam / (cam.max() + K.epsilon())
    return cam


def nonce_str():
    """
    Nonce string generator, based on timestamp.
    
    :return: unique string, based on timestamp
    """
    return str(time.time())


def clean(path):
    """
    Delete all files into a directory, recursively, excluding directories themselves.

    :param path: directory to clean
    :return: None
    """
    
    if os.path.exists(path):
        for file in os.listdir(path):
            filename = path + "/" + file
            if os.path.isdir(filename):
                clean(filename)
            else:
                os.remove(filename)


def deprocess_image(x):
    """
    Copy the nd_array received as input and representing an image, reduce its dimension to three so it can be printed
    at screen, and then do same normalization as in:
    https://github.com/fchollet/keras/blob/master/examples/conv_filter_visualization.py.

    This function is based on code from https://github.com/eclique/keras-gradcam/blob/master/gradcam_vgg.ipynb.

    :param x: nd_array representing the generated image
    :return: processed nd_array, which could be used in e.g. imshow
    """
    
    x = x.copy()
    if np.ndim(x) > 3:
        x = np.squeeze(x)
    
    # normalize tensor: center on 0., ensure std is 0.25
    x -= x.mean()
    x /= (x.std() + K.epsilon())
    x *= 0.25
    
    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)
    
    # convert to RGB array
    x *= 255
    if K.image_data_format() == 'th':
        x = x.transpose((1, 2, 0))
    x = np.clip(x, 0, 255).astype('uint8')
    return x


def custom_preprocessing(mean, std_dev=None):
    """
    Wrapper for custom pre-processing function.
    
    Return custom image pre-processing function that takes nd_array of input image to pre-process and then perform the
    pre-processing: mean subtraction from input nd_array and, if provided, division by standard deviation.
    
    :param mean: custom training dataset's mean, provided by user
    :param std_dev: custom training dataset's standard deviation, provided by user
    :return: custom pre-processing function
    """
    
    def preprocess(x):
        x = x.copy()
    
        x -= mean
        if std_dev is not None:
            x /= (std_dev + K.epsilon())
        return x
    
    return preprocess


def custom_decode_predictions(n_classes, class_index_file, preds, top=1):
    """
    Decode the prediction of a custom classification model, using a custom CLASS INDEX loaded through a JSON file.
    This function is based on decode_predictions function for ImageNet models, defined in keras.applications.
    
    :param n_classes: number of classes available for classification
    :param class_index_file: path to JSON file containing custom class mapping
    :param preds: nd_array containing a batch of predictions
    :param top: integer, how many predictions to return
    :return: list of lists of top class prediction tuples `(class_name, class_description, score)`.
        One list of tuples per sample in batch input.
        
    Raises: ValueError
    """
    
    if n_classes < 2:
        raise ValueError("Classification cannot be done with n_classes=" + str(n_classes) + ".")

    if len(preds.shape) != 2:
        raise ValueError("'preds' must be a batch of predictions (i.e. a 2D array of shape (samples, n_classes)).\n" +
                         "Found array with shape: " + str(preds.shape) + ".")
    
    if preds.shape[1] != n_classes:
        raise ValueError("'preds' must be a batch of predictions (i.e. a 2D array of shape (samples, n_classes)).\n" +
                         "Found array with shape: " + str(preds.shape) + ", but n_classes: " + str(n_classes) + ".")

    if top < 1 or top > n_classes:
        raise ValueError("Invalid 'top' request (" + str(top) + ").\n" +
                         "Correct values for 'top' are from 1 to n_classes (" + str(n_classes) + ").")

    if not os.path.exists(class_index_file):
        raise ValueError("Custom class index file not found in '" + class_index_file + ".")

    with open(class_index_file) as f:
        custom_class_index = json.load(f)

    results = []
    for pred in preds:
        top_indices = pred.argsort()[-top:][::-1]
        result = [tuple(custom_class_index[str(i)]) + (pred[i],) for i in top_indices]
        result.sort(key=lambda x: x[1], reverse=True)
        results.append(result)
    return results


@bp.route('/', methods=('GET', 'POST'))
def index():
    """
    Main endpoint, home of the application.
    When called, it makes sure that the directory's structure needed for inputs and outputs exists and is empty.
    
    :return: html index template
    """
    
    if not os.path.exists(INPUT_IMAGES_PATH):
        os.makedirs(INPUT_IMAGES_PATH)

    if not os.path.exists(INPUT_INDEXES_PATH):
        os.makedirs(INPUT_INDEXES_PATH)

    if not os.path.exists(INPUT_MODELS_PATH):
        os.makedirs(INPUT_MODELS_PATH)

    if not os.path.exists(OUTPUT_IMAGES_PATH):
        os.makedirs(OUTPUT_IMAGES_PATH)

    #clean(INPUT_PATH)
    #clean(OUTPUT_PATH)
    
    return render_template('main/index.html')

def move_image(backend_path, frontend_path, img_name):
    # Set the source and destination paths
    source_path = backend_path + img_name

    # # Move the file
    # shutil.move(source_path, destination_path)

    #Copy image file for debug
    shutil.copy(source_path, os.path.join(frontend_path, img_name))

@bp.route('/upload_images', methods=('POST',))
@cross_origin(origins="*")
def upload_images():
    """
    Endpoint for ajax request for uploading input image/s.

    :return: Response containing loaded image/s filename or error
    """

    images = request.files.getlist("images")
    for image in images:
        if not image:
            response = Response(0, "aborted")
            return jsonify(response.toJSON())
    
    try:
        images = [images_UploadSet.save(image, name=image.filename + "_" + nonce_str() + ".") for image in images]
        response = Response(1, "success", content=images)
    except flask_uploads.UploadNotAllowed:
        response = Response(0, "error", "Invalid input image/s.")
    
    current_path = os.getcwd()
    print(current_path)
    for image in images:   
        print(image)  
        try:    
            move_image(INPUT_IMAGES_PATH, FRONTEND_INPUT_PATH, image)
        except:
            pass
    return jsonify(response.toJSON())


@bp.route('/upload_custom_model', methods=('POST',))
def upload_custom_model():
    """
    Endpoint for ajax request for uploading custom model HDF5 file.

    :return: Response containing loaded custom model filename or error
    """
    
    requested_file = request.files["custom_model"]
    if not requested_file:
        response = Response(0, "aborted")
        return jsonify(response.toJSON())
    
    try:
        custom_model = models_UploadSet.save(requested_file, name=requested_file.filename + "_" + nonce_str() + ".")
        response = Response(1, "success", content=custom_model)
    except flask_uploads.UploadNotAllowed:
        response = Response(0, "error", "Invalid custom model file.")

    return jsonify(response.toJSON())


@bp.route('/upload_custom_class_index', methods=('POST',))
def upload_custom_class_index():
    """
    Endpoint for ajax request for uploading custom class index JSON file.

    :return: Response containing loaded custom class index filename or error
    """
    
    requested_file = request.files["custom_class_index"]
    if not requested_file:
        response = Response(0, "aborted")
        return jsonify(response.toJSON())
    
    try:
        custom_class_index = class_indexes_UploadSet.save(requested_file, name="custom_class_index" + nonce_str() + ".")
        response = Response(1, "success", content=custom_class_index)
    except flask_uploads.UploadNotAllowed:
        response = Response(0, "error", "Invalid custom class index file.")

    return jsonify(response.toJSON())


@bp.route('/upload_training_images', methods=('POST',))
def upload_training_images():
    """
    Endpoint for ajax request for uploading folder with training images for visualization method that requires it.

    :return: Response containing loaded image/s filename or error
    """

    training_images = request.files.getlist("training_images")
    for training_image in training_images:
        if not training_image:
            response = Response(0, "aborted")
            return jsonify(response.toJSON())

    try:
        training_images = [images_UploadSet.save(training_image, name="training/image" + nonce_str() + ".")
                           for training_image in training_images]
        response = Response(1, "success", content=training_images)
    except flask_uploads.UploadNotAllowed:
        response = Response(0, "error", "Invalid training image/s.")

    return jsonify(response.toJSON())


@bp.route('/download_layers', methods=('POST',))
def download_layers():
    """
    Endpoint for ajax request for downloading layers of a loaded model or a predefined Keras one.

    :return: Response containing array of layers' names or error
    """

    # clean the Tensorflow graph in order to use another instance of the model each time visualize() is called
    K.clear_session()
    
    # read JSON data
    json_request = request.get_json()
    custom_model_selected = json_request["custom_model_selected"]
    print(json_request)
    
    # load model
    if custom_model_selected:
        custom_model_file = json_request["model_name"]
        if not custom_model_file:
            response = Response(0, "error", "Invalid custom model file.")
            return jsonify(response.toJSON())
    
        try:
            model = load_model(models_UploadSet.path(custom_model_file))
        except Exception as e:
            response = Response(0, "error", "Unable to load custom model: " + str(e))
            return jsonify(response.toJSON())
    else:
        predefined_model = json_request["model_name"]

        if predefined_model == "VGG16":
            model = applications.vgg16.VGG16()
        elif predefined_model == "VGG19":
            model = applications.vgg19.VGG19()
        elif predefined_model == "ResNet50":
            model = applications.resnet50.ResNet50()
        elif predefined_model == "Xception":
            model = applications.xception.Xception()
        elif predefined_model == "InceptionV3":
            model = applications.inception_v3.InceptionV3()
        elif predefined_model == "InceptionResNetV2":
            model = applications.inception_resnet_v2.InceptionResNetV2()
        elif predefined_model == "DenseNet121":
            model = applications.densenet.DenseNet121()
        elif predefined_model == "DenseNet169":
            model = applications.densenet.DenseNet169()
        elif predefined_model == "DenseNet201":
            model = applications.densenet.DenseNet201()
        elif predefined_model == "NASNetLarge":
            model = applications.nasnet.NASNetLarge()
        elif predefined_model == "NASNetMobile":
            model = applications.nasnet.NASNetMobile()
        elif predefined_model == "MobileNet":
            model = applications.mobilenet.MobileNet()
        elif predefined_model == "MobileNetV2":
            model = applications.mobilenet_v2.MobileNetV2()
        else:
            response = Response(0, "error", "Invalid pre-defined model selection.")
            return jsonify(response.toJSON())

    layers = model.layers
    
    # consider only layers available for selected visualization method
    method = json_request["method"]
    if method == "gradcam" or method == "guided_gradcam":
        for i in range(len(layers)):
            if len(layers[i].output.shape) < 4:
                break
        
        # remove all layers after last convolutional one
        for j in range(len(layers) - i):
            layers.pop(i)

    layers.pop(0)  # remove input layer that cannot be visualized in any case
    
    # save names of all available layers
    layer_names = [layer.name for layer in layers]

    response = Response(1, "success", content=layer_names)
    return jsonify(response.toJSON())


@bp.route('/download_classes', methods=('POST',))
def download_classes():
    """
    Endpoint for ajax request for downloading classes predicted by a loaded model.
    If the request is about a predefined Keras model, it returns the 1000's ImageNet classes.

    :return: Response containing array of classes or error
    """
    
    # read JSON data
    json_request = request.get_json()
    
    custom_model_selected = json_request["custom_model_selected"]
    if custom_model_selected == "False":
        custom_model_selected = False
    elif custom_model_selected == "True":
        custom_model_selected = True
    if custom_model_selected:
        custom_class_index = json_request["custom_class_index"]
        custom_n_classes = json_request["custom_n_classes"]
        
        if not custom_class_index or not custom_n_classes:
            response = Response(0, "error", "Unable to get classes.")
            return jsonify(response.toJSON())
        
        custom_n_classes = int(custom_n_classes)
        if custom_n_classes < 2:
            response = Response(0, "error", "Number of classes must be at least 2.")
            return jsonify(response.toJSON())
        
        class_index_file = class_indexes_UploadSet.path(custom_class_index)
        if not class_index_file:
            response = Response(0, "error", "Invalid class index file.")
            return jsonify(response.toJSON())
        
        n_classes = custom_n_classes
    else:
        print("Start Download")
        # from keras_applications/imagenet_utils.py
        class_index_path = ('https://s3.amazonaws.com/deep-learning-models/'
                            'image-models/imagenet_class_index.json')
        
        class_index_file = get_file('imagenet_class_index.json',
                                    class_index_path,
                                    cache_subdir='models',
                                    file_hash='c2c37ea517e94d9795004a39431a14cb')
        n_classes = 1000
    
    with open(class_index_file) as f:
        class_index = json.load(f)
    
    try:
        classes = [str(i) + " - " + class_index[str(i)][-1] for i in range(n_classes)]
    except Exception:
        response = Response(0, "error", "Invalid class uploading.")
        return jsonify(response.toJSON())
    
    response = Response(1, "success", content=classes)
    return jsonify(response.toJSON())


def reduce(x):
    """
    Reduce an ndarray into a single value by summing its elements.
    
    :param x: ndarray to be reduced
    :return: sum of the elements of x
    """
    return np.sum(x)


def get_inputs(reason, json_request):
    """
    Read input data from client json_request and treat them differently if they are for Visualization or for Suggestion.

    :param reason: either "visualization" or "suggestion"
    :param json_request: JSON request array
    :return: saved input data
    """
    
    # clean the Tensorflow graph in order to use another instance of the model each time visualize() is called
    K.clear_session()

    print(json_request)
    
    # check input images
    input_images = json_request["images"]  # list of input image/s filename
    if not input_images or len(input_images) <= 0:
        response = [Response(0, "error", "Invalid input image/s.")]
        return response
    
    if reason == "suggestion":
        if len(input_images) <= 1:
            response = [Response(0, "error", "More than a single image is needed for receiving any suggestion.")]
            return response
    
    # model selection
    custom_model_selected = json_request["custom_model_selected"]  # flag
    n_models = int(json_request["n_models"])
    if not n_models:
        response = [Response(0, "error", "Model not selected.")]
        return response

    models = []
    preprocesses = []
    custom_class_index_file = ""
    decode_predictions = []
    if (type(custom_model_selected) == str):
        custom_model_selected = False
    print(type(custom_model_selected))
    if custom_model_selected:
        print("In Here")
        final_custom_models = json_request["final_custom_models"]
        if len(final_custom_models) != n_models:
            response = [Response(0, "error",
                                 "Invalid number of custom model/s loaded.\nExpected: " + str(n_models) + ".")]
            return response
    
        custom_models = []
        custom_means = []
        custom_std_devs = []
        for custom_model in final_custom_models:
            custom_models.append(custom_model["name"])
            custom_means.append(custom_model["means"])
            custom_std_devs.append(custom_model["std_devs"])
    
        # load custom class index file and custom n_classes if user want to use custom classification mapping
        custom_class_index = json_request["custom_class_index"]
        n_classes = json_request["custom_n_classes"]
        if custom_class_index and not n_classes:
            response = [Response(0, "error", "Number of classes must be at least 2.")]
            return response
        elif not custom_class_index and n_classes:
            response = [Response(0, "error", "Invalid class index file.")]
            return response
        elif custom_class_index and n_classes:
            # user insert both custom_class_index file and n_classes, custom classification mapping can be used
            n_classes = int(n_classes)
            if n_classes < 2:
                response = [Response(0, "error", "Number of classes must be at least 2.")]
                return response
        
            custom_class_index_file = class_indexes_UploadSet.path(custom_class_index)
            if not custom_class_index_file:
                response = [Response(0, "error", "Invalid class index file.")]
                return response
        else:
            # user doesn't load both custom_class_index file and n_classes, custom classification mapping is not used
            # n_classes is set to default 0
            n_classes = 0
    
        # load custom models
        for i in range(n_models):
            custom_model_file = custom_models[i]
            if not custom_model_file:
                response = [Response(0, "error", "Invalid custom model file.")]
                return response
        
            try:
                models.append(load_model(models_UploadSet.path(custom_model_file)))
            except Exception as e:
                response = [Response(0, "error", "Unable to load custom model: " + str(e))]
                return response
        
            # load custom mean and standard deviation for custom pre-preprocessing
            if not custom_means[i] and not custom_std_devs[i]:
                # use ImageNet pre-processing function
                preprocesses.append(applications.vgg16.preprocess_input)
            elif not custom_means[i] and custom_std_devs[i]:
                response = [Response(0, "error", "For custom pre-processing, training dataset's mean is required.")]
                return response
            elif custom_means[i] and not custom_std_devs[i]:
                custom_mean = np.asarray(custom_means[i]).astype(float)
            
                if np.any(custom_mean < 0):
                    response = [Response(0, "error", "Mean can't be negative.")]
                    return response
            
                # use pre-processing = mean subtraction
                preprocesses.append(custom_preprocessing(custom_mean))
            else:
                custom_mean = np.asarray(custom_means[i]).astype(float)
                custom_std_dev = np.asarray(custom_std_devs[i]).astype(float)
            
                if np.any(custom_mean < 0):
                    response = [Response(0, "error", "Mean can't be negative.")]
                    return response
            
                if np.any(custom_std_dev < 0):
                    response = [Response(0, "error", "Standard deviation can't be negative.")]
                    return response
            
                # use pre-processing = mean subtraction then division by standard deviation
                preprocesses.append(custom_preprocessing(custom_mean, custom_std_dev))
    else:
        predefined_models = json_request["predefined_models"]
        if len(predefined_models) != n_models:
            response = [Response(0, "error", "Invalid number of predefined model/s.\nExpected: " + str(n_models) + ".")]
            return response
    
        # classification mapping is given by ImageNet, n_classes set to default 1000
        n_classes = 1000
    
        # load predefined models
        for i in range(n_models):
            predefined_model = predefined_models[i]
        
            if predefined_model == "VGG16":
                models.append(applications.vgg16.VGG16())
                preprocesses.append(applications.vgg16.preprocess_input)
                decode_predictions.append(applications.vgg16.decode_predictions)
            elif predefined_model == "VGG19":
                models.append(applications.vgg19.VGG19())
                preprocesses.append(applications.vgg19.preprocess_input)
                decode_predictions.append(applications.vgg19.decode_predictions)
            elif predefined_model == "ResNet50":
                models.append(applications.resnet50.ResNet50())
                preprocesses.append(applications.resnet50.preprocess_input)
                decode_predictions.append(applications.resnet50.decode_predictions)
            elif predefined_model == "Xception":
                models.append(applications.xception.Xception())
                preprocesses.append(applications.xception.preprocess_input)
                decode_predictions.append(applications.xception.decode_predictions)
            elif predefined_model == "InceptionV3":
                models.append(applications.inception_v3.InceptionV3())
                preprocesses.append(applications.inception_v3.preprocess_input)
                decode_predictions.append(applications.inception_v3.decode_predictions)
            elif predefined_model == "InceptionResNetV2":
                models.append(applications.inception_resnet_v2.InceptionResNetV2())
                preprocesses.append(applications.inception_resnet_v2.preprocess_input)
                decode_predictions.append(applications.inception_resnet_v2.decode_predictions)
            elif predefined_model == "DenseNet121":
                models.append(applications.densenet.DenseNet121())
                preprocesses.append(applications.densenet.preprocess_input)
                decode_predictions.append(applications.densenet.decode_predictions)
            elif predefined_model == "DenseNet169":
                models.append(applications.densenet.DenseNet169())
                preprocesses.append(applications.densenet.preprocess_input)
                decode_predictions.append(applications.densenet.decode_predictions)
            elif predefined_model == "DenseNet201":
                models.append(applications.densenet.DenseNet201())
                preprocesses.append(applications.densenet.preprocess_input)
                decode_predictions.append(applications.densenet.decode_predictions)
            elif predefined_model == "NASNetLarge":
                models.append(applications.nasnet.NASNetLarge())
                preprocesses.append(applications.nasnet.preprocess_input)
                decode_predictions.append(applications.nasnet.decode_predictions)
            elif predefined_model == "NASNetMobile":
                models.append(applications.nasnet.NASNetMobile())
                preprocesses.append(applications.nasnet.preprocess_input)
                decode_predictions.append(applications.nasnet.decode_predictions)
            elif predefined_model == "MobileNet":
                alpha = json_request["mobilenet_alpha"]  # optional
                depth_multiplier = json_request["mobilenet_depth_multiplier"]  # optional
                dropout = json_request["mobilenet_dropout"]  # optional
            
                if not alpha:
                    alpha = 1.0
            
                if not depth_multiplier:
                    depth_multiplier = 1
            
                if not dropout:
                    dropout = 1e-3
            
                try:
                    models.append(applications.mobilenet.MobileNet(alpha=float(alpha),
                                                                   depth_multiplier=int(depth_multiplier),
                                                                   dropout=float(dropout)))
                except Exception as e:
                    response = [Response(0, "error", str(e) + "\nSee log for more information.")]
                    app.logger.exception(e, exc_info=True)
                    return response
            
                preprocesses.append(applications.mobilenet.preprocess_input)
                decode_predictions.append(applications.mobilenet.decode_predictions)
            elif predefined_model == "MobileNetV2":
                alpha = json_request["mobilenetv2_alpha"]  # optional
            
                if not alpha:
                    alpha = 1.0
            
                try:
                    models.append(applications.mobilenet_v2.MobileNetV2(alpha=float(alpha)))
                except Exception as e:
                    response = [Response(0, "error", str(e) + "\nSee log for more information.")]
                    app.logger.exception(e, exc_info=True)
                    return response
            
                preprocesses.append(applications.mobilenet_v2.preprocess_input)
                decode_predictions.append(applications.mobilenet_v2.decode_predictions)
            else:
                response = [Response(0, "error", "Invalid pre-defined model selection.")]
                return response

    # save input shape required for the model (input images will be resized to this dimensions)
    model_input_shapes = []
    for i in range(n_models):
        model_input_shapes.append(models[i].input_shape[1:-1])

    # method selection
    method = json_request["method"]  # name of the chosen visualization method
    if reason == "visualization":
        if not method or method not in methods:
            response = [Response(0, "error", "Invalid visualization method.")]
            return response
        
    # otherwise, there are some suggestions that work also without having selected a visualization method
    
    return input_images, n_models, models, preprocesses, model_input_shapes, n_classes, custom_model_selected, \
        custom_class_index_file, decode_predictions, method


def predict_classes_for_image(input_image, custom_model_selected, model, model_input_shape, preprocess, n_classes,
                              custom_class_index_file, decode_predictions=None):
    """
    Make the output predictions for an input image using a custom/predefined model.
    
    :param input_image: filename of the input image
    :param custom_model_selected: flag for custom/predefined model selection
    :param model: Keras model to use
    :param model_input_shape: model input shape, in order to properly resize the input image
    :param preprocess: pre-processing function to be used on the image before giving it in input to the model
    :param n_classes: number of classes for the model
    :param custom_class_index_file: path to CLASS_INDEX to associate a prediction to a class name
        (only for custom models)
    :param decode_predictions: function used to decode the predictions nd_array obtained by the model
    :return: (nd_array of input image, nd_array of input image after pre-processing, classification_mappings,
        output_class_id, prediction nd_array)
    """
    
    # load image resizing it properly for model input shape
    image = preprocessing.image.load_img(images_UploadSet.path(input_image),
                                         target_size=model_input_shape)
    
    # transform input image in nd_array with right dimensions
    nd_image = preprocessing.image.img_to_array(image)
    nd_image = np.expand_dims(nd_image, axis=0)
    
    # do the preprocessing
    preprocessed_input = preprocess(nd_image)
    
    # save information about classification
    classification_mappings = []
    output_class_id = ""
    preds = [[]]
    
    if n_classes > 0:
        # save predictions for print them into the visualization
        preds = model.predict(preprocessed_input)
        
        # save class ids for all predictions
        classes = np.argsort(preds[0])[-n_classes:][::-1]
        
        # save id of output class (max probability)
        output_class_id = classes[0]  # np.argmax(preds)
        
        if custom_model_selected:
            decoded_preds = custom_decode_predictions(n_classes, custom_class_index_file,
                                                      preds, top=n_classes)[0]
            for c, dp in zip(classes, decoded_preds):
                # dp[0] contains class name, dp[1] contains score
                classification_mappings.append(ClassificationMapping(int(c), str(dp[0]),
                                                                     round(float(dp[1]), 2)).toJSON())
        else:
            decoded_preds = decode_predictions(preds, top=n_classes)[0]
            for c, dp in zip(classes, decoded_preds):
                # dp[1] contains class name, dp[2] contains score
                classification_mappings.append(ClassificationMapping(int(c), str(dp[1]),
                                                                     round(float(dp[2]), 2)).toJSON())
            
    return image, preprocessed_input, classification_mappings, output_class_id, preds[0]


def select_layer(reason, json_request, model, method):
    """
    Select a layer inside a model, in order to have a resulting model ending at the selected layer.
    
    :param reason: either "visualization" or "suggestion", because of different controls needed
    :param json_request: JSON request from the client containing input for selecting the layer (layer_name)
    :param model: model on which to select the layer
    :param method: method to be used in case of reason == "visualization", because of different controls needed
    :return: (selected layer name, selected layer index, truncated model at selected layer)
    """
    
    layer_name = json_request["layer_name"]
    layer_index = None
    
    if layer_name == "":
        if reason == "visualization":
            # gradcam is not included in iNNvestigate, custom implementation always need layer index
            if method == "gradcam" or method == "guided_gradcam":
                layers = model.layers
                i = 2  # initialization, I suppose any model at a certain point as a flat layer
                for i in range(len(layers)):
                    if len(layers[i].output.shape) < 4:
                        break
        
                # # default is last convolutional layer
                # if isinstance(layers[i - 1], _Conv):
                #     layer_index = i - 1
                # else:
                #     # layers[i-1] is assumed to be the pooling layer after the convolutional one
                #     layer_index = i - 2
                layer_index = i - 1
                    
            if method != "gradcam":
                # strip softmax layer: ok for all method with layer selection except for gradcam
                try:
                    model = innvestigate.utils.model_wo_softmax(model)
                except:
                    response = [Response(0, "error", "Softmax activation not found in last layer of input model/s.\n"
                                                     "Please specify a layer for visualization.\n")]
                    return response
    else:
        layer_names = [layer.name for layer in model.layers]
        if layer_name not in layer_names:
            response = [Response(0, "error", "Invalid layer selection.\n")]
            return response
        
        for idx, layer in enumerate(model.layers):
            if layer.name == layer_name:
                layer_index = idx
                break
        
        try:
            # transform the model in order to finish at the selected layer, then visualize that last layer
            model = Model(inputs=model.input, outputs=model.get_layer(index=layer_index).output)
        except Exception as e:
            response = [Response(0, "error", str(e))]
            return response

    # if selected layer is not flat, make it flat in order to properly select neurons in it
    if len(model.output.shape) > 2:
        model_output = GlobalMaxPooling2D()(model.output)
        model = Model(inputs=model.input, outputs=model_output)
        
    return layer_name, layer_index, model


def select_neuron(json_request):
    """
    Used to select which neuron/s to use inside a layer of a model.
    
    :param json_request: JSON request containing input from the client
    :return: (neuron_selection_mode, number of neurons corresponding to max activations in case
        neuron_selection_mode == "max_activations" or neuron_index in case neuron_selection_mode == "index"
    """
    
    neuron_selection_mode = json_request["neuron_selection_mode"]
    if not neuron_selection_mode or neuron_selection_mode not in neuron_selection_modes:
        response = [Response(0, "error", "Invalid neuron selection mode.\nValid values are: " +
                             str(neuron_selection_modes))]
        return response

    n_max_activations = json_request["n_max_activations"]
    neuron_index = json_request["neuron_index"]
    if neuron_selection_mode == "max_activations":
        if not n_max_activations or n_max_activations not in valid_n_max_activations:
            response = [Response(0, "error", "Invalid number of max activations selected.\nValid values are: " +
                                 str(valid_n_max_activations))]
            return response
        
        n_max_activations = int(n_max_activations)
    
    elif neuron_selection_mode == "index":
        if neuron_index == "":
            response = [Response(0, "error", "Neuron selection mode is '" + neuron_selection_mode +
                                 "': you must provide neuron index.")]
            return response
        
        neuron_index = int(neuron_index)
        
    return neuron_selection_mode, n_max_activations, neuron_index
    

@bp.route('/suggest', methods=('POST',))
@cross_origin(origins="*")
def suggest():
    """
    Endpoint for ajax request for producing requested suggestion.
    
    :return: Response containing all data related to the computed suggestion
    """

    # read JSON data
    json_request = request.get_json()
    
    response = get_inputs("suggestion", json_request)
    if len(response) == 1:
        return jsonify(response[0].toJSON())
    
    input_images, n_models, models, preprocesses, model_input_shapes, n_classes, custom_model_selected, \
        custom_class_index_file, decode_predictions, method = response

    input_images_belong_to_same_class = json_request["input_images_belong_to_same_class"]
    input_images_class_id = json_request["input_images_class_id"]
    
    suggestions = []
    n = len(input_images)
    try:
        if n_models > 1:
            suggestion = Suggestion("max_models_predictions_mean_score")

            n = len(input_images)
            models_top_pred_mean_probs = []
            for i in range(n):
                predictions = []
                for m in range(n_models):
                    if custom_model_selected:
                        _, _, classification_mappings, output_class_id, _ = \
                            predict_classes_for_image(input_images[i], custom_model_selected, models[m],
                                                      model_input_shapes[m], preprocesses[m], n_classes,
                                                      custom_class_index_file)
                    else:
                        _, _, classification_mappings, output_class_id, _ = \
                            predict_classes_for_image(input_images[i], custom_model_selected, models[m],
                                                      model_input_shapes[m], preprocesses[m], n_classes,
                                                      custom_class_index_file, decode_predictions[m])
        
                    predictions.append(classification_mappings)
    
                models_groups = []
                models_group = {
                    "input_image_id": i,
                    "n_models": 1,
                    "models_index": [0],
                    "class_id": predictions[0][0]["classId"],
                    "class_name": predictions[0][0]["className"],
                    "mean_prob": predictions[0][0]["score"]
                }
                models_groups.append(models_group)
    
                for m in range(1, n_models):
                    found_group_with_same_pred = False
                    for group in models_groups:
                        if predictions[m][0]["classId"] == group["class_id"]:
                            found_group_with_same_pred = True
                            group["n_models"] += 1
                            group["models_index"].append(m)
                            group["mean_prob"] += predictions[m][0]["score"]
        
                    if not found_group_with_same_pred:
                        models_group = {
                            "input_image_id": i,
                            "n_models": 1,
                            "models_index": [m],
                            "class_id": predictions[m][0]["classId"],
                            "class_name": predictions[m][0]["className"],
                            "mean_prob": predictions[m][0]["score"]
                        }
                        models_groups.append(models_group)
    
                for group in models_groups:
                    group["mean_prob"] /= group["n_models"]
    
                models_top_pred_mean_probs.append(models_groups)

            models_top_pred_mean_probs.sort(key=lambda x: x[0]["mean_prob"], reverse=True)
            models_top_pred_mean_probs.sort(key=len)

            ordered_input_images = []
            for i in range(n):
                ordered_input_images.append(input_images[models_top_pred_mean_probs[i][0]["input_image_id"]])

            suggestion.input_images = ordered_input_images
            suggestion.models_top_pred_mean_probs = models_top_pred_mean_probs
            
            suggestions.append(suggestion.toJSON())
            
            # second suggestion available if user select more than one model
            if input_images_belong_to_same_class:
                if not input_images_class_id:
                    response = Response(0, "error", "Invalid class for input images.")
                    return jsonify(response.toJSON())
                input_images_class_id = int(input_images_class_id)
                input_images_class_name = ""
                
                suggestion = Suggestion("correct_models_predictions")
                
                ground_truth_vector = np.zeros(n_classes)
                ground_truth_vector[input_images_class_id] = 1
                
                n_models_correct = []
                models_correct = []
                models_mean_distance_from_input_class = []
                for i in range(n):
                    n_correct = 0
                    models_correct_for_img = []
                    models_mean_distance_for_img = 0
                    for m in range(n_models):
                        if custom_model_selected:
                            _, _, classification_mappings, output_class_id, preds = \
                                predict_classes_for_image(input_images[i], custom_model_selected, models[m],
                                                          model_input_shapes[m], preprocesses[m], n_classes,
                                                          custom_class_index_file)
                        else:
                            _, _, classification_mappings, output_class_id, preds = \
                                predict_classes_for_image(input_images[i], custom_model_selected, models[m],
                                                          model_input_shapes[m], preprocesses[m], n_classes,
                                                          custom_class_index_file, decode_predictions[m])
                        
                        # save input images class name related to input images class id inserted by the user
                        if input_images_class_name == "":
                            for l in range(n_classes):
                                if classification_mappings[l]["classId"] == input_images_class_id:
                                    input_images_class_name = classification_mappings[l]["className"]
                                    break
                                    
                        if output_class_id == input_images_class_id:
                            n_correct += 1
                            models_correct_for_img.append(m)

                        distance = cosine(preds, ground_truth_vector)
                        models_mean_distance_for_img += distance
                            
                    n_models_correct.append(n_correct)
                    models_correct.append(models_correct_for_img)
                    
                    models_mean_distance_for_img /= n_models
                    models_mean_distance_from_input_class.append(models_mean_distance_for_img)
                
                ordered_indexes = np.argsort(n_models_correct)[::-1]
                models_correct.sort(key=len, reverse=True)
                
                ordered_input_images = []
                ordered_models_mean_distance_from_input_class = []
                for i in range(n):
                    ordered_input_images.append(input_images[ordered_indexes[i]])
                    ordered_models_mean_distance_from_input_class.append(
                        models_mean_distance_from_input_class[ordered_indexes[i]])
                    
                suggestion.input_images = ordered_input_images
                suggestion.models_correct = models_correct
                suggestion.models_mean_distance_from_input_class = ordered_models_mean_distance_from_input_class
                suggestion.input_images_class_name = input_images_class_name
                
                suggestions.append(suggestion.toJSON())
                    
        else:
            if not method:
                response = Response(0, "error", "Choose a visualization method for receiving suggestion for input "
                                                "images based on the only model you loaded/choose (or load/choose "
                                                "more than one model for other suggestions than don't need any other "
                                                "information)")
                return jsonify(response.toJSON())
            
            suggestion = Suggestion()
            m = 0
            model_predictions = []

            # get selected layer
            response = select_layer("suggestion", json_request, models[m], method)
            if len(response) == 1:
                return jsonify(response[0].toJSON())
            
            original_model = models[m]
            layer_name, layer_index, models[m] = response

            # get selected layer activations (needed by both suggestion 2 and 3)
            images_activations = []
            for i in range(n):
                if custom_model_selected:
                    _, preprocessed_input, classification_mappings, _, _ = \
                        predict_classes_for_image(input_images[i], custom_model_selected, original_model,
                                                  model_input_shapes[m], preprocesses[m], n_classes,
                                                  custom_class_index_file)
                else:
                    _, preprocessed_input, classification_mappings, _, _ = \
                        predict_classes_for_image(input_images[i], custom_model_selected, original_model,
                                                  model_input_shapes[m], preprocesses[m], n_classes,
                                                  custom_class_index_file, decode_predictions[m])
    
                model_predictions.append(classification_mappings)
    
                activations_function = K.function([models[m].input], [models[m].output])
                activations = activations_function([preprocessed_input])[0]
                images_activations.append(activations)
                
            # if user select a visualization method that provide neuron selection, check this first:
            # if neuron_selection_mode == index => show suggestion based on that neuron
            ordered_indexes = []
            neuron_selection = True
            if method == "gradient" or method == "gradient.baseline" or method == "smoothgrad" \
                    or method == "deconvnet" or method == "guided_backprop" or method == "pattern.net":
                
                response = select_neuron(json_request)
                if len(response) == 1:
                    return jsonify(response[0].toJSON())
    
                neuron_selection_mode, n_max_activations, neuron_index = response
                
                if neuron_selection_mode == "index":
                    suggestion.kind = "max_neuron_activation"

                    neuron_activations = []
                    for i in range(n):
                        neuron_activations.append(float(images_activations[i][0][neuron_index]))
                    
                    ordered_indexes = np.argsort(neuron_activations)[::-1]
                    neuron_activations.sort(reverse=True)

                    suggestion.neuron_activations = neuron_activations
                    
                else:
                    neuron_selection = False
            
            else:
                neuron_selection = False
                    
            if not neuron_selection:
                # if neuron index was not specified (because visualization method do not provide it or because user did
                # not insert it) => use just selected layer to show suggestions
                suggestion.kind = "max_layer_mean_activation"
    
                layer_mean_activations = []
                for i in range(n):
                    layer_mean_activations.append(float(np.mean(images_activations[i][0])))
    
                ordered_indexes = np.argsort(layer_mean_activations)[::-1]
                layer_mean_activations.sort(reverse=True)

                suggestion.layer_mean_activations = layer_mean_activations

            ordered_input_images = []
            ordered_model_predictions = []
            for i in range(n):
                ordered_input_images.append(input_images[ordered_indexes[i]])
                ordered_model_predictions.append(model_predictions[ordered_indexes[i]])

            suggestion.input_images = ordered_input_images
            suggestion.model_predictions = ordered_model_predictions
            
            suggestions.append(suggestion.toJSON())
            
    except Exception as e:
        response = Response(0, "error", str(e) + "\nSee log for more information.")
        app.logger.exception(e, exc_info=True)
        return jsonify(response.toJSON())

    app.logger.info("'Suggest' - suggestions: " + str(suggestions))

    response = Response(1, "success", content=suggestions)
    return jsonify(response.toJSON())


@bp.route('/visualize', methods=('POST',))
@cross_origin(origins="*")
def visualize():
    """
    Endpoint for ajax request for producing requested visualization.
    
    :return: Response containing all data related to the computed visualization
    """
    
    # read JSON data
    json_request = request.get_json()
    if (json_request['custom_model_selected'] == "False"):
        json_request['custom_model_selected'] = False
    elif(json_request['custom_model_selected'] == "True"):
        json_request['custom_model_selected'] = True

    response = get_inputs("visualization", json_request)
    if len(response) == 1:
        return jsonify(response[0].toJSON())
    
    input_images, n_models, models, preprocesses, model_input_shapes, n_classes, custom_model_selected, \
        custom_class_index_file, decode_predictions, method = response
    
    try:
        print("Enter")
        visualizations = []
        m = 0
    
        for model in models:
            # save copy of original models
            original_model = Model(inputs=model.input, outputs=model.output)

            # get selected layer
            response = select_layer("visualization", json_request, model, method)
            if len(response) == 1:
                return jsonify(response[0].toJSON())
            
            layer_name, layer_index, model = response
            
            # save original model layers' names
            # then change them in order to guarantee unique names for iNNvestigate analysis
            layer_names = []
            for i, layer in enumerate(original_model.layers):
                layer_names.append(layer.name)
                layer._name = 'layer_' + str(i)
                
            # create analyzers using additional parameters specific for selected visualization method and
            # using layer selection and neuron/class selection
            innvestigate_neuron_selection_mode = ""
            if method == "input":
                analyzer = innvestigate.create_analyzer(method, original_model, allow_lambda_layers=True)
    
            elif method == "random":
                stddev = json_request["stddev"]  # optional
            
                if not stddev:
                    stddev = 1  # as default in iNNvestigate
                else:
                    stddev = float(stddev)
    
                analyzer = innvestigate.create_analyzer(method, original_model, stddev=stddev, allow_lambda_layers=True)
    
            # all other methods have layer selection
            else:
                # methods with neuron selection
                if method == "gradient" or method == "gradient.baseline" or method == "smoothgrad" \
                        or method == "deconvnet" or method == "guided_backprop" or method == "pattern.net":

                    print("test gradient enter")
                    # get selected neuron
                    response = select_neuron(json_request)
                    print(response)
                    if len(response) == 1:
                        return jsonify(response[0].toJSON())

                    neuron_selection_mode, n_max_activations, neuron_index = response

                    # in any case, we use iNNvestigate neuron_selection_mode parameter equal to "index"
                    innvestigate_neuron_selection_mode = "index"
                    
                    if method.startswith("gradient"):
                        print(json_request["gradient_postprocess"])
                        postprocess = json_request["gradient_postprocess"]  # optional
            
                        if not postprocess:
                            postprocess = "abs"
                        elif postprocess not in valid_gradient_postprocesses:
                            response = Response(0, "error", "Invalid postprocess.\nValid values are: " +
                                                str(valid_gradient_postprocesses))
                            return jsonify(response.toJSON())
            
                        if postprocess == "None":
                            postprocess = None
                            
                        print("Gradient analyzer enter")
                        analyzer = innvestigate.create_analyzer(
                            method, model,
                            neuron_selection_mode=innvestigate_neuron_selection_mode,
                            postprocess=postprocess,
                            allow_lambda_layers=True
                        )
    
                    elif method == "smoothgrad":
                        augment_by_n = json_request["augment_by_n"]  # optional
                        postprocess = json_request["smoothgrad_postprocess"]  # optional
            
                        if not augment_by_n:
                            augment_by_n = 32  # iNNvestigate default = 64
                        else:
                            augment_by_n = int(augment_by_n)
            
                        if not postprocess:
                            postprocess = "abs"
                        elif postprocess not in valid_gradient_postprocesses:
                            response = Response(0, "error", "Invalid postprocess.\nValid values are: " +
                                                str(valid_gradient_postprocesses))
                            return jsonify(response.toJSON())
            
                        if postprocess == "None":
                            postprocess = None
        
                        # allow_lambda_layers=True parameter not working for this visualization method on iNNvestigate
                        try:
                            analyzer = innvestigate.create_analyzer(
                                method, model,
                                neuron_selection_mode=innvestigate_neuron_selection_mode,
                                augment_by_n=augment_by_n,
                                postprocess=postprocess
                            )
                        except innvestigate.NotAnalyzeableModelException:
                            response = Response(0, "error",
                                                "Unable to use this method with models containing lambda layers.")
                            return jsonify(response.toJSON())
        
                    elif method == "pattern.net":
                        # check training images
                        training_images = json_request["training_images"]  # list of training image/s filename
                        if not training_images or len(training_images) <= 0:
                            response = Response(0, "error", "Invalid training image/s.")
                            return jsonify(response.toJSON())

                        analyzer = innvestigate.create_analyzer(
                            method, model,
                            neuron_selection_mode=innvestigate_neuron_selection_mode,
                            allow_lambda_layers=True
                        )
                        
                    else:
                        analyzer = innvestigate.create_analyzer(
                            method, model,
                            neuron_selection_mode=innvestigate_neuron_selection_mode,
                            allow_lambda_layers=True
                        )
                
                # methods with class selection
                elif method == "input_t_gradient" or method == "integrated_gradients" or method.startswith("lrp") \
                        or method.startswith("deep_taylor") or method == "deep_lift.wrapper" \
                        or method == "pattern.attribution" or method == "gradcam" or method == "guided_gradcam":
                    
                    # save class selection
                    class_id = json_request["class_id"]
                    print(class_id)
                    if not class_id:
                        response = Response(0, "error", "Invalid class selection.")
                        return jsonify(response.toJSON())
                    class_id = int(class_id)
    
                    innvestigate_neuron_selection_mode = "index"
                    
                    if layer_name != "":
                        original_model_output_activations_for_class = original_model.output[0, class_id]
                        original_model_selected_layer_output_activations = \
                            original_model.get_layer(index=layer_index).output
    
                        grads = K.gradients(original_model_output_activations_for_class,
                                            original_model_selected_layer_output_activations)[0]
                        
                        gradient_function = K.function([original_model.input],
                                                       [original_model_selected_layer_output_activations, grads])
                    
                    if method == "guided_gradcam":
                        # guided_gradcam method is not included in iNNvestigate:
                        # it needs gradcam computation (custom) and guided_backprop computation (from iNNvestigate)
                        analyzer = innvestigate.create_analyzer("guided_backprop", model,
                                                                neuron_selection_mode="all",
                                                                allow_lambda_layers=True)
    
                    elif method == "integrated_gradients":
                        steps = json_request["steps"]  # optional
                        postprocess = json_request["integrated_gradients_postprocess"]  # optional
            
                        if not steps:
                            steps = 32  # iNNvestigate default = 64
                        else:
                            steps = int(steps)
            
                        if not postprocess:
                            postprocess = "abs"
                        elif postprocess not in valid_gradient_postprocesses:
                            response = Response(0, "error", "Invalid postprocess.\nValid values are: " +
                                                str(valid_gradient_postprocesses))
                            return jsonify(response.toJSON())
            
                        if postprocess == "None":
                            postprocess = None

                        # allow_lambda_layers=True parameter not working for this visualization method on iNNvestigate
                        try:
                            analyzer = innvestigate.create_analyzer(
                                method, model,
                                neuron_selection_mode=innvestigate_neuron_selection_mode,
                                steps=steps,
                                postprocess=postprocess
                            )
                        except innvestigate.NotAnalyzeableModelException:
                            response = Response(0, "error",
                                                "Unable to use this method with models containing lambda layers.")
                            return jsonify(response.toJSON())
                        
                    elif method == "lrp":
                        rule = json_request["rule"]
            
                        if not rule or rule not in lrp_rules:
                            response = Response(0, "error", "Invalid rule for lrp.\nValid values are: " +
                                                str(lrp_rules))
                            return jsonify(response.toJSON())
            
                        if rule == "Epsilon":
                            epsilon = json_request["epsilon"]  # optional
                            epsilon_bias = json_request["lrp_epsilon_bias"]  # optional
            
                            if not epsilon:
                                epsilon = 1e-7  # as default in iNNvestigate
                            else:
                                epsilon = float(epsilon)
            
                            if epsilon_bias == "False":
                                epsilon_bias = False
                            else:
                                epsilon_bias = True
            
                            analyzer = innvestigate.create_analyzer(
                                "lrp.epsilon", model,
                                neuron_selection_mode=innvestigate_neuron_selection_mode,
                                epsilon=epsilon,
                                bias=epsilon_bias,
                                allow_lambda_layers=True
                            )
            
                        elif rule == "EpsilonIgnoreBias":
                            epsilon = json_request["lrp_epsilon_IB_sequential_preset_AB_epsilon"]  # optional
            
                            if not epsilon:
                                epsilon = 1e-7  # as default in iNNvestigate
                            else:
                                epsilon = float(epsilon)
            
                            analyzer = innvestigate.create_analyzer(
                                "lrp.epsilon_IB", model,
                                neuron_selection_mode=innvestigate_neuron_selection_mode,
                                epsilon=epsilon,
                                allow_lambda_layers=True
                            )
            
                        elif rule == "AlphaBeta":
                            alpha = json_request["alpha"]
                            beta = json_request["beta"]
                            alpha_beta_bias = json_request["lrp_alpha_beta_bias"]  # optional
            
                            if not alpha or not beta:
                                response = Response(0, "error", "LRP with rule " + rule + " requires alpha and beta.")
                                return jsonify(response.toJSON())
            
                            alpha = float(alpha)
                            beta = float(beta)
                            if alpha_beta_bias == "False":
                                alpha_beta_bias = False
                            else:
                                alpha_beta_bias = True
            
                            analyzer = innvestigate.create_analyzer(
                                "lrp.alpha_beta", model,
                                neuron_selection_mode=innvestigate_neuron_selection_mode,
                                alpha=alpha,
                                beta=beta,
                                bias=alpha_beta_bias,
                                allow_lambda_layers=True
                            )
            
                        elif rule == "AlphaBetaIgnoreBias":
                            alpha = json_request["alpha"]
                            beta = json_request["beta"]
            
                            if not alpha or not beta:
                                response = Response(0, "error", "LRP with rule " + rule + " requires alpha and beta.")
                                return jsonify(response.toJSON())
            
                            alpha = float(alpha)
                            beta = float(beta)
            
                            analyzer = innvestigate.create_analyzer(
                                "lrp.alpha_beta", model,
                                neuron_selection_mode=innvestigate_neuron_selection_mode,
                                alpha=alpha,
                                beta=beta,
                                bias=False,
                                allow_lambda_layers=True
                            )
            
                        elif rule == "Bounded":
                            low = json_request["low"]  # optional
                            high = json_request["high"]  # optional
            
                            if not low:
                                low = -1  # as default in iNNvestigate
                            else:
                                low = int(low)
            
                            if not high:
                                high = 1  # as default in iNNvestigate
                            else:
                                high = int(high)
            
                            if low > high:
                                response = Response(0, "error", "Low bound can't be higher than high bound.")
                                return jsonify(response.toJSON())
            
                            analyzer = innvestigate.create_analyzer(
                                "lrp", model,
                                neuron_selection_mode=innvestigate_neuron_selection_mode,
                                rule=rule,
                                input_layer_rule=(low, high),
                                allow_lambda_layers=True
                            )
            
                        else:
                            analyzer = innvestigate.create_analyzer(
                                method, model,
                                neuron_selection_mode=innvestigate_neuron_selection_mode,
                                rule=rule,
                                allow_lambda_layers=True
                            )
            
                    elif method == "lrp.epsilon":
                        epsilon = json_request["epsilon"]  # optional
                        bias = json_request["lrp_epsilon_bias"]  # optional
            
                        if not epsilon:
                            epsilon = 1e-7  # as default in iNNvestigate
                        else:
                            epsilon = float(epsilon)
            
                        if bias == "False":
                            bias = False
                        else:
                            bias = True
            
                        analyzer = innvestigate.create_analyzer(
                            method, model,
                            neuron_selection_mode=innvestigate_neuron_selection_mode,
                            epsilon=epsilon,
                            bias=bias,
                            allow_lambda_layers=True
                        )
            
                    elif method == "lrp.epsilon_IB" \
                            or method == "lrp.sequential_preset_a" or method == "lrp.sequential_preset_b":
                        
                        epsilon = json_request["lrp_epsilon_IB_sequential_preset_AB_epsilon"]  # optional
            
                        if not epsilon:
                            if method == "lrp.epsilon_IB":
                                epsilon = 1e-7  # as default in iNNvestigate for lrp.epsilon_IB
                            else:
                                epsilon = 1e-1  # as default in iNNvestigate for lrp.sequential_preset_a and b
                        else:
                            epsilon = float(epsilon)
            
                        analyzer = innvestigate.create_analyzer(
                            method, model,
                            neuron_selection_mode=innvestigate_neuron_selection_mode,
                            epsilon=epsilon,
                            allow_lambda_layers=True
                        )
            
                    elif method == "lrp.alpha_beta":
                        alpha = json_request["alpha"]
                        beta = json_request["beta"]
                        bias = json_request["lrp_alpha_beta_bias"]  # optional
            
                        if not alpha or not beta:
                            response = Response(0, "error", method + " requires alpha and beta.")
                            return jsonify(response.toJSON())
            
                        alpha = float(alpha)
                        beta = float(beta)
                        if bias == "False":
                            bias = False
                        else:
                            bias = True
            
                        analyzer = innvestigate.create_analyzer(
                            method, model,
                            neuron_selection_mode=innvestigate_neuron_selection_mode,
                            alpha=alpha,
                            beta=beta,
                            bias=bias,
                            allow_lambda_layers=True
                        )
            
                    elif method == "deep_taylor.bounded":
                        deep_taylor_low = json_request["deep_taylor_low"]
                        deep_taylor_high = json_request["deep_taylor_high"]
            
                        if not deep_taylor_low or not deep_taylor_high:
                            response = Response(0, "error", method + " requires high and low bounds.")
                            return jsonify(response.toJSON())
            
                        deep_taylor_low = float(deep_taylor_low)
                        deep_taylor_high = float(deep_taylor_high)
            
                        if deep_taylor_low > deep_taylor_high:
                            response = Response(0, "error", "Low bound can't be higher than high bound.")
                            return jsonify(response.toJSON())
            
                        analyzer = innvestigate.create_analyzer(
                            method, model,
                            neuron_selection_mode=innvestigate_neuron_selection_mode,
                            low=deep_taylor_low,
                            high=deep_taylor_high,
                            allow_lambda_layers=True
                        )
            
                    elif method == "pattern.attribution":
                        # check training images
                        training_images = json_request["training_images"]  # list of training image/s filename
                        if not training_images or len(training_images) <= 0:
                            response = Response(0, "error", "Invalid training image/s.")
                            return jsonify(response.toJSON())

                        analyzer = innvestigate.create_analyzer(
                            method, model,
                            neuron_selection_mode=innvestigate_neuron_selection_mode,
                            allow_lambda_layers=True
                        )
                        
                    elif method != "gradcam":
                        analyzer = innvestigate.create_analyzer(
                            method, model,
                            neuron_selection_mode=innvestigate_neuron_selection_mode,
                            allow_lambda_layers=True
                        )
                
                else:
                    analyzer = innvestigate.create_analyzer(method, model, allow_lambda_layers=True)
            
            # fit iNNvestigate analyzer for visualization methods that need to be fitted
            if method.startswith("pattern"):
                x_train_list = []
                for training_image_filename in training_images:
                    # load image resizing it properly for model input shape
                    training_image = preprocessing.image.load_img(
                        images_UploadSet.path(training_image_filename),
                        target_size=model_input_shapes[m])
        
                    # transform training image in nd_array with right dimensions
                    nd_training_image = preprocessing.image.img_to_array(training_image)
                    nd_training_image = np.expand_dims(nd_training_image, axis=0)
        
                    # do the preprocessing
                    preprocessed_training_image = preprocesses[m](nd_training_image)
        
                    # save preprocessed training image into x_train_list
                    x_train_list.append(preprocessed_training_image[0])
    
                x_train = np.array(x_train_list)
    
                # fit analyzer using training images
                analyzer.fit(x_train)
                
            # produce the required visualization for all input images
            n = len(input_images)
            output_images = []
            predictions = []
            max_activations_neuron_indexes_per_image = []
            for i in range(n):
                if custom_model_selected:
                    image, preprocessed_input, classification_mappings, output_class_id, _ = \
                        predict_classes_for_image(input_images[i], custom_model_selected, original_model,
                                                  model_input_shapes[m], preprocesses[m], n_classes,
                                                  custom_class_index_file)
                else:
                    image, preprocessed_input, classification_mappings, output_class_id, _ = \
                        predict_classes_for_image(input_images[i], custom_model_selected, original_model,
                                                  model_input_shapes[m], preprocesses[m], n_classes,
                                                  custom_class_index_file, decode_predictions[m])
                
                if len(classification_mappings) > 0:
                    predictions.append(classification_mappings)
    
                # do the analysis
                neuron_indexes = []
                # methods with neuron selection
                if method == "gradient" or method == "gradient.baseline" or method == "smoothgrad" \
                        or method == "deconvnet" or method == "guided_backprop" or method == "pattern.net":
                    
                    # for these methods, it is possible to have multiple results in one figure
                    result = []
                    if neuron_selection_mode == "index":
                        result.append(analyzer.analyze(preprocessed_input, neuron_index))
    
                    elif neuron_selection_mode == "max_activations":
                        activations_function = K.function([model.input], [model.output])
                        activations = activations_function([preprocessed_input])[0]
                        neuron_indexes = np.argsort(activations)[0, 0:n_max_activations]
                        
                        # convert neuron_indexes in order to be serialized for passing it to the html for visualization
                        neuron_indexes = list(neuron_indexes)
                        for n in range(len(neuron_indexes)):
                            neuron_indexes[n] = int(neuron_indexes[n])
                            result.append(analyzer.analyze(preprocessed_input, neuron_selection=neuron_indexes[n]))
                
                # custom analysis (not provided by iNNvestigate) with class selection
                elif method == "gradcam":
                    if len(model.get_layer(index=layer_index).output.shape) < 4:
                        response = Response(0, "error", "Invalid layer for method " + method +
                                            ".\nOnly layer operating on convolutional filters are accepted.")
                        return jsonify(response.toJSON())

                    if class_id == -1:
                        result = grad_cam(original_model, preprocessed_input, output_class_id, layer_index,
                                          model_input_shapes[m])
                    else:
                        result = grad_cam(original_model, preprocessed_input, class_id, layer_index,
                                          model_input_shapes[m])

                # custom analysis (not provided by iNNvestigate) with class selection
                elif method == "guided_gradcam":
                    if len(model.get_layer(index=layer_index).output.shape) < 4:
                        response = Response(0, "error", "Invalid layer for method " + method +
                                            ".\nOnly layer operating on convolutional filters are accepted.")
                        return jsonify(response.toJSON())

                    gb = analyzer.analyze(preprocessed_input)

                    if class_id == -1:
                        gradcam = grad_cam(original_model, preprocessed_input, output_class_id, layer_index,
                                           model_input_shapes[m])
                    else:
                        gradcam = grad_cam(original_model, preprocessed_input, class_id, layer_index,
                                           model_input_shapes[m])
    
                    result = gb * gradcam[..., np.newaxis]
    
                # methods with class selection
                elif method == "input_t_gradient" or method == "integrated_gradients" or method.startswith("lrp") \
                        or method.startswith("deep_taylor") or method == "deep_lift.wrapper" \
                        or method == "pattern.attribution":
                    
                    # do the analysis on neurons related to previously selected class
                    if layer_name != "":
                        # compute gradients for selected layer via gradient_function defined above
                        _, grads_val = gradient_function([preprocessed_input])
                        
                        # reduce gradient to one dimension if selected layer is not a flat one
                        if len(grads_val.shape) > 2:
                            summary_grads = [reduce(grads_val[0, 0:, 0:, conv_filter])
                                             for conv_filter in range(grads_val.shape[-1])]
                        else:
                            summary_grads = grads_val[0]
        
                        # compute mean value for gradients' amplitudes
                        summary_grads_amps = np.abs(summary_grads)
                        avg_amp = (np.amax(summary_grads_amps) + np.amin(summary_grads_amps)) / 2
    
                        # select neurons corresponding to gradients having amplitude grater than the mean value
                        neuron_indexes = [j for j in range(grads_val.shape[-1])
                                          if summary_grads_amps[j] > avg_amp]
        
                        neuron_indexes_len = len(neuron_indexes)
                        if neuron_indexes_len == 0:
                            # if no neuron was selected, compute output using neuron with max amplitude gradient
                            result = analyzer.analyze(preprocessed_input, np.argmax(summary_grads_amps))
                        else:
                            # if just one neuron is selected, compute output using this neuron
                            result = analyzer.analyze(preprocessed_input, neuron_selection=neuron_indexes[0])
                            
                            # if more than one neuron has been selected, compute a mean output visualization among all
                            # the output visualizations computed by using each of the selected neurons once at a time
                            if neuron_indexes_len > 1:
                                for j in range(neuron_indexes_len - 1):
                                    result += analyzer.analyze(preprocessed_input, neuron_indexes[j+1])
    
                                result = np.divide(result, neuron_indexes_len)
                    else:
                        if class_id == -1:
                            result = analyzer.analyze(preprocessed_input, output_class_id)
                        else:
                            result = analyzer.analyze(preprocessed_input, class_id)
                         
                else:
                    result = analyzer.analyze(preprocessed_input)
                    
                # produce output image
                nonce = nonce_str()
                
                if method.startswith("gradient") or method == "smoothgrad" or method == "deconvnet" \
                        or method == "guided_backprop" or method == "pattern.net":
                    
                    print("HI")
                    n_result = len(result)
                    if n_result == 1:
                        nr = nc = 1
                        result = result[0]
    
                        fig, axs = plt.subplots()
                        fig.subplots_adjust(wspace=0, hspace=0.04)

                        axs.axis("off")
                        if method.startswith("gradient") or method == "smoothgrad":
                            result = result.sum(axis=np.argmax(np.asarray(result.shape) == 3))
                            result /= (np.max(np.abs(result)) + K.epsilon())
        
                            img = axs.imshow(result[0], cmap="gray", clim=(0, 1), interpolation="nearest")
    
                        else:
                            axs.imshow(deprocess_image(result[0]))
                    
                    elif n_result == 2:
                        nr = 1
                        nc = 2
                        
                        fig, axs = plt.subplots(1, 2)
                        fig.subplots_adjust(wspace=0.04, hspace=0.04)
    
                        for l in range(n_result):
                            axs[l].axis("off")
                            
                            if method.startswith("gradient") or method == "smoothgrad":
                                result[l] = result[l].sum(axis=np.argmax(np.asarray(result[l].shape) == 3))
                                result[l] /= (np.max(np.abs(result[l])) + K.epsilon())
            
                                img = axs[l].imshow(result[l][0], cmap="gray", clim=(0, 1), interpolation="nearest")
        
                            else:
                                axs[l].imshow(deprocess_image(result[l][0]))
                        
                    else:
                        nr = nc = int(sqrt(n_result))
                        square = nr * nc
                        is_n_result_square = square == n_result
                        if not is_n_result_square:
                            nr += 1
                            nc += 1 
                            square = nr * nc
                        
                        fig, axs = plt.subplots(nr, nc)
                        fig.subplots_adjust(wspace=0, hspace=0.04)
                        
                        r = c = 0
                        for l in range(n_result):
                            axs[r, c].axis("off")
                            
                            if method.startswith("gradient") or method == "smoothgrad":
                                result[l] = result[l].sum(axis=np.argmax(np.asarray(result[l].shape) == 3))
                                result[l] /= (np.max(np.abs(result[l])) + K.epsilon())
        
                                img = axs[r, c].imshow(result[l][0], cmap="gray", clim=(0, 1), interpolation="nearest")
                            
                            else:
                                axs[r, c].imshow(deprocess_image(result[l][0]))
                                
                            c += 1
                            if c == nc:
                                r += 1
                                c = 0
                        
                        if not is_n_result_square:
                            for l in range(n_result, square):
                                axs[r, c].imshow(np.zeros(result[0][0].shape), cmap="gray")
                                axs[r, c].axis("off")
                                
                                c += 1
                                if c == nc:
                                    r += 1
                                    c = 0

                    if method.startswith("gradient") or method == "smoothgrad":
                        # save image without legend colorbar
                        output_image = "visualization_" + str(m) + "_image_" + str(i) + "_" + nonce + "_nocolorbar.jpg"
                        fig.savefig(OUTPUT_IMAGES_PATH + output_image, bbox_inches="tight", pad_inches=0)
    
                        # add colorbar for image saved for visualization on browser
                        if nr < nc:
                            fig.colorbar(img, ax=axs, orientation="horizontal")
                        else:
                            fig.colorbar(img, ax=axs)

                    # save produced image and its name
                    output_image = "visualization_" + str(m) + "_image_" + str(i) + "_" + nonce + ".jpg"

                    fig.savefig(OUTPUT_IMAGES_PATH + output_image, bbox_inches="tight", pad_inches=0)
                    
                else:
                    fig, axs = plt.subplots()
                    fig.subplots_adjust(wspace=0, hspace=0.04)
                    
                    axs.axis("off")
                    
                    if method == "input" or method == "random" or method == "guided_gradcam":
                        axs.imshow(deprocess_image(result[0]))
    
                    elif method == "gradcam":
                        axs.imshow(image)
                        img = axs.imshow(result, cmap="jet", clim=(0, 1), alpha=0.5)
    
                        # save image without legend colorbar
                        output_image = "visualization_" + str(m) + "_image_" + str(i) + "_" + nonce + "_nocolorbar.jpg"
                        fig.savefig(OUTPUT_IMAGES_PATH + output_image, bbox_inches="tight", pad_inches=0)
    
                        # add colorbar for image saved for visualization on browser
                        fig.colorbar(img, ax=axs)
        
                    elif method == "input_t_gradient" or method == "integrated_gradients" or method.startswith("lrp")\
                            or method.startswith("deep_taylor") or method == "deep_lift.wrapper" \
                            or method == "pattern.attribution":
                        axs.imshow(image)
        
                        # Aggregate along color channels and normalize to [-1, 1]
                        result = result.sum(axis=np.argmax(np.asarray(result.shape) == 3))
                        result /= (np.max(np.abs(result)) + K.epsilon())
        
                        img = axs.imshow(result[0], cmap="seismic", clim=(-1, 1), alpha=0.8)
                        
                        # save image without legend colorbar
                        output_image = "visualization_" + str(m) + "_image_" + str(i) + "_" + nonce + "_nocolorbar.jpg"
                        fig.savefig(OUTPUT_IMAGES_PATH + output_image, bbox_inches="tight", pad_inches=0)
    
                        # add colorbar for image saved for visualization on browser
                        fig.colorbar(img, ax=axs)
        
                    else:
                        axs.imshow(deprocess_image(result[0]))

                    print("HERE")
                    # save produced image and its name
                    output_image = "visualization_" + str(m) + "_image_" + str(i) + "_" + nonce + ".jpg"
                    
                    fig.savefig(OUTPUT_IMAGES_PATH + output_image, bbox_inches="tight", pad_inches=0)
                    
                plt.cla()
                plt.clf()
                
                output_images.append(output_image)
                max_activations_neuron_indexes_per_image.append(list(neuron_indexes))
                
            # restore original model layers' names for model plotting
            for i, layer in enumerate(original_model.layers):
                layer._name = layer_names[i]
    
            # original_model.summary(print_fn=app.logger.debug)
            # model.summary(print_fn=app.logger.debug)

            visualizations.append(Visualization(output_images, max_activations_neuron_indexes_per_image, predictions)
                                  .toJSON())
            m += 1

    except Exception as e:
        response = Response(0, "error", str(e) + "\nSee log for more information.")
        app.logger.exception(e, exc_info=True)
        return jsonify(response.toJSON())

    app.logger.info("'Visualize' - input images: " + str(input_images) + " - visualizations: " + str(visualizations))

    response = Response(1, "success", content={"input_images": input_images, "visualizations": visualizations})
    for image in visualizations[0]['output_images']:
        try:
            move_image(OUTPUT_IMAGES_PATH, FRONTEND_OUTPUT_PATH, image)
        except:
            pass
    return jsonify(response.toJSON())