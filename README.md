
# Churn_Prediction_Project
# Customer Churn Prediction Model

## Overview

This Customer Churn Prediction Model was developed for SunBase Data. It is designed to predict customer churn based on a dataset provided by the company. The model achieved its best results using the Bagging Classifier, but it also includes other models that were explored during development.

## Dataset

The dataset provided by SunBase Data contains information about customers, including features such as age, gender, location, subscription length, monthly bill, total usage, and more. This dataset was used to train and test the machine learning models.

## Models

The project includes the following machine learning models:

1. Bagging Classifier (Best Performer): Achieved the highest accuracy in predicting customer churn.

2. Decision Tree Classifier: Explored as a baseline model.

3. Random Forest Classifier: Tested for its ensemble learning capabilities.

## Usage

To use this Customer Churn Prediction Model, follow these steps:

1. Ensure you have Python and the required libraries installed.

2. Clone this repository to your local machine.

3. Run the `main.py` script to load the Flask web application.

4. Access the web application in your browser to input customer information and receive churn predictions.

## Model Evaluation

The models were evaluated using various metrics, including accuracy, precision, recall, and F1-score. The Bagging Classifier consistently outperformed other models in predicting customer churn.

## Technologies Used

- Python
- Scikit-learn
- Flask
- Pandas
- NumPy
- HTML/CSS
- Jupyter Notebook (for data exploration and model development)

## Project Structure

- `main.py`: Flask web application for making predictions.
- `churn_prediction.ipynb`: Jupyter Notebook containing data preprocessing and model development.
- `model1.sav`: Saved Bagging Classifier model.
- `templates/`: HTML templates for the web application.
- `static/`: Static files for the web application.

![Screenshot 2023-10-20 204650](https://github.com/Jainamrockk/Churn_Prediction_Project/assets/67656502/91bff769-a886-4e36-8401-16aed3a6f45f)

## Future Enhancements

- Incorporate more data sources and features for even better predictions.
- Explore deep learning techniques for more complex models.
- Implement real-time data updates and predictions.

## Credits

This project was created by Jainam Jain for SunBase Data.

Special thanks to the SunBase Data team for providing the dataset and support during development.

If you have any questions or suggestions, please contact jainamjain2002@gmail.com

Happy predicting!
