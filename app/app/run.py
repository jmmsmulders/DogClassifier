import re
import random
import re
from glob import glob
import numpy as np
from keras.layers import GlobalAveragePooling2D
from keras.layers import Dense
from keras.models import Sequential
from keras.utils import np_utils
from tensorflow.keras.preprocessing import image  
from sklearn.datasets import load_files   
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from PIL import Image
from io import BytesIO
import io

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Generate model
Resnet50_model = Sequential()
# Convert features of Resnet-model to vectors
Resnet50_model.add(GlobalAveragePooling2D(input_shape=(7,7,2048)))
# Add a dense-layer to generate an output per category
Resnet50_model.add(Dense(133, activation='softmax'))
Resnet50_model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
Resnet50_model.load_weights('../../models/model.hdf5')

# Load Train Targets / Images
def load_dataset(path):
    data = load_files(path)
    dog_files = np.array(data['filenames'])
    dog_targets = np_utils.to_categorical(np.array(data['target']), 133)
    return dog_files, dog_targets

# load train, test, and validation datasets
train_files, train_targets = load_dataset('../../app/app/static/dogs')

# load list of dog names
dog_names = [item[20:-1] for item in sorted(glob("../../app/app/static/dogs/*/"))]


def extract_Resnet50(tensor):
    """
    Function that extract features of the pre-trained ResNet50 model for classifying Dogs

    Args:
        tensor: numpy array of the image to predict for
    
    Output:
        Prediction of the pre-trained model
    """
    return ResNet50(weights='imagenet', include_top=False).predict(preprocess_input(tensor), verbose=0)


### returns "True" if a dog is detected in the image stored at img_path
def dog_detector(img):
    # Load model
    ResNet50_model = ResNet50(weights='imagenet')
    # define ResNet50 model
    img = preprocess_input(img)
    prediction = np.argmax(ResNet50_model.predict(img, verbose=0))
    return ((prediction <= 268) & (prediction >= 151)) 


def path_to_tensor(img_path):
    '''
    Function that reshapes an image to to tensors in the correct shape for the CNN-model

    Args:
        img_path: Location of the image
    
    Output:
        Numpy array of the image in the shape (224, 224, 3)
    '''
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=(224, 224))
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)


def find_random_image_of_dog(idx):
    '''
    Function that takes an idx corresponding to the label of a dog breed
    Then finds a random image of the breed in the training files and return the path
    
    Args:
        idx: Index of the dog breed as an integer within the range of 0 to 133
    
    Output:
        img_path: Location of the selected image
    '''
    idx = np.where(train_targets[:,idx] == 1)
    sample_img = random.choice(idx[0])
    img_path = train_files[sample_img]
    return ".." + img_path[13:]

def format_dog_name(idx):
    '''
    Function that formats the output-string of the classification algorithm to a more readable format
    
    Args:
        idx: Index of the dog breed as an integer within the range of 0 to 133
    
    Output:
        name: Formatted string of the dog name
    '''
    name = dog_names[idx]
    name = name.split('.')[1]
    name = re.sub('[^0-9a-zA-Z]+', ' ', name)
    return name


def find_top3(top_3):
    """
    Function that returns the top_3 most likely dog-classifcation for the selected image 
    and also returns a random image of that breed

    Args:
        top_3: Dictionary, with the index of the breed on keys and the classification percentage as an item
    
    Output:
        name_0, name_1, name_2: Name of the Dog breed with the top 3 most resemblance
        percentage_0, percentage_1, percentage_2: Result of the algorithm corresponding to the percentage of classifcation
        random_img_0, random_img_1, random_img_2: Random image of the predicted breed 
    """
    top_3 = {k: v for k, v in sorted(top_3.items(), reverse=True, key=lambda item: item[1])}

    name_0 = format_dog_name(list(top_3.keys())[0])
    percentage_0 = np.round(list(top_3.values())[0]*100, 2)
    random_img_0 = find_random_image_of_dog(list(top_3.keys())[0])

    name_1 = format_dog_name(list(top_3.keys())[1])
    percentage_1 = np.round(list(top_3.values())[1]*100, 2)
    random_img_1 = find_random_image_of_dog(list(top_3.keys())[1])

    name_2 = format_dog_name(list(top_3.keys())[2])
    percentage_2 = np.round(list(top_3.values())[2]*100, 2)
    random_img_2 = find_random_image_of_dog(list(top_3.keys())[2])

    return name_0, percentage_0, random_img_0, \
    name_1, percentage_1, random_img_1, \
    name_2, percentage_2, random_img_2

# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')

def index():
    # render web page with plotly graphs
    return render_template('master.html')
    

@app.route('/classify', methods=['POST'])       
def Resnet50_predict_breed(n=3):
    '''
    Function that makes a prediction on image to determine the dog-breed, it also returns the top-n of most similar breeds
    
    Args:
        img_path: Location of the image to predict for
        n: top-n to return
        
    Output:
        dog_name: Name of the most likely predicted dog-breed as a string
        top_n: Dictionary with the 3 most likely breeds, and their prediction percentages
    '''
    # Get the image file from the request
    image_file = request.files['image_file']
    img = path_to_tensor(io.BytesIO(image_file.read()))

    if dog_detector(img) == True:
        s = 'Hello there doggie! \nLook at what a good boy you are!'            
    else:
        s = "This does not like a dog!\n Nevertheless, let's see what breed of dog this picture resembles most" 

    # extract bottleneck features
    bottleneck_feature = extract_Resnet50(img)

    # obtain predicted vector
    predicted_vector = Resnet50_model.predict(bottleneck_feature, verbose=0)    

    # Top-n Predictions
    idx = np.argsort(predicted_vector, axis=1)[:,-n:]
    top_3 = {}

    for i in idx[0]:
        top_3[i] = predicted_vector[:,i][0]

    name_0, percentage_0, random_img_0, \
    name_1, percentage_1, random_img_1, \
    name_2, percentage_2, random_img_2 = find_top3(top_3)    
 
    return jsonify({'name_0': name_0,
                    'name_1': name_1,
                    'name_2': name_2,
                    'percentage_0': percentage_0,
                    'percentage_1': percentage_1,
                    'percentage_2': percentage_2,
                    'random_img_0': random_img_0,
                    'random_img_1': random_img_1,
                    'random_img_2': random_img_2,
                    'human_dog': s})

def main():
    app.run(host='0.0.0.0', port=3000, debug=True)

if __name__ == '__main__':
    main()