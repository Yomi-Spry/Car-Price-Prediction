# Car-Price-Prediction

This repository hosts a machine learning project focused on predicting car prices using a variety of features. The project utilizes a dataset consisting of car listings and employs a regression model to estimate the price of a car based on its attributes.

## Project Structure
The repository is structured as follows:

- `sample.csv`:This directory houses the dataset used for training and evaluation purposes. It comprises car listings with pertinent attributes, including make, model, year, mileage, and other relevant information.

- `trained_model.sav`: This file is a serialized version of the trained machine learning model for car price prediction.

- `cleaning_pipe.sav`: This file contains the serialized version of an encoder object used to transform categorical variables into a numerical format suitable for the machine learning model.

- `auto_pipe.sav`: This file contains the serialized version of a preprocessing object used to tranform and engineer the variables for the machine learning model.

- `Trader.py`: This directory contains a web application for car price prediction. Users can input car details, and the application will return an estimated price based on the trained model.

- 
`requirements.txt`: This file lists all the required Python libraries and dependencies to run the project.

## Usage
To use this project, follow these steps:

1. Install the necessary dependencies by running the following command:
    ```
        pip install -r requirements.txt 
    ```
2. **Launching The App**
    Run the `Trader.py` script from your command line to start the car price prediction web application.

3. **Conveinient URL Access**
    As the application starts, a URL will be displayed in the command line interface. Copy the URL and paste it directly into your browser to access the application.

  
