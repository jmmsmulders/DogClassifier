# DogClassifier
 
This repo contains code and logic in order to run a Flask-App, which makes it possible to make a classification of the dogbreed of any images it is fed as input. 
And is used as my submission for the Capstone Project of the Udacity Data Scientist Course.

This repo consists of the following parts:
- Notebook: Notebook used in the course to create a CNN Classification Algorithm based on Transfer-learning in order to classify a dog-breed
- Models: The trained model in order to Classify with
- App: The code needed to run the Flask App
 
## Installation
In order to run all the code in this repo you need at least need to have a python installation and the python packages `keras`, `glob`, `tensorflow (version == 1.4.46)`, `sklearn`, `PIL`, `io`

## Usage
In order to run the Flask App, navigate tot the app-folder with `cd app` then run the following command in an interpreter: `python run.py`

This results in the creation of the following web-app, which you can access by following the http link in your terminal:
![image](https://user-images.githubusercontent.com/118716035/223985943-e8870190-a860-486f-bddc-96e4445dd18e.png)

Where you can manually input an image on your PC, by pressing the "Classify Dog Breed" button a top-3 prediction on breeds should be given for the input image:
![image](https://user-images.githubusercontent.com/118716035/223986349-6d3b6da0-158b-454a-987c-09ed2e2c98b2.png)


## Contributing
Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are greatly appreciated.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement". Don't forget to give the project a star! Thanks again!

## Fork the Project
- Create your Feature Branch (git checkout -b feature/AmazingFeature)
- Commit your Changes (git commit -m 'Add some AmazingFeature')
- Push to the Branch (git push origin feature/AmazingFeature)
- Open a Pull Request

## Contact
Joep Smulders - (https://www.linkedin.com/in/joep-smulders-200203117/) - smulders.jmm@gmail.com
Project Link: (https://github.com/jmmsmulders/ThreeInARow)
