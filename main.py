import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingClassifier
from sklearn import metrics
from flask import Flask, request, render_template
import zipfile
import pickle

# Name of the zip archive
zip_archive = 'C:\\Users\\Jainam\\PycharmProjects\\Churn_Prediction_Project\\model.zip'

# Initialize a variable to store the model
model = None

with zipfile.ZipFile(zip_archive, 'r') as zipf:
    model_filename = 'model1.sav'

    # Extract the model file
    with zipf.open(model_filename) as model_file:
        # Load the model from the extracted file
        model = pickle.load(model_file)
app = Flask(__name__)

# Load the dataset
df_1 = pd.read_excel("C:\\Users\\Jainam\\PycharmProjects\\Churn_Prediction_Project\\templates\\customer_churn_large_dataset.xlsx")

@app.route("/")
def loadPage():
    return render_template('home.html', query="")

@app.route("/", methods=['POST'])
def predict():
    # Get input values from the form
    inputQuery1 = request.form['query1']
    inputQuery2 = request.form['query2']
    inputQuery3 = request.form['query3']
    inputQuery4 = request.form['query4']
    inputQuery5 = request.form['query5']
    inputQuery6 = request.form['query6']
    
    # Load the pre-trained model
    # model = pickle.load(open("C:\\Users\\Jainam\\PycharmProjects\\Churn_Prediction_Project\\model1.sav", "rb"))
    
    # Create a new DataFrame with user input
    data = [[inputQuery1, inputQuery2, inputQuery3, inputQuery4, inputQuery5, inputQuery6]]
    new_df = pd.DataFrame(data, columns=['Age', 'Gender', 'Location', 'Subscription_Length_Months', 'Monthly_Bill', 'Total_Usage_GB'])
    
    # Merge the new data with the original dataset
    df_2 = pd.concat([df_1, new_df], ignore_index=True)
    
    # Data cleaning and preprocessing
    df_2['Subscription_Length_Months'] = pd.to_numeric(df_2['Subscription_Length_Months'], errors='coerce')
    df_2['Age'] = pd.to_numeric(df_2['Age'], errors='coerce')
    labels = ["{0} - {1}".format(i, i+11) for i in range(1, df_2['Subscription_Length_Months'].max(), 12)]
    df_2['tenure_group'] = pd.cut(df_2.Subscription_Length_Months, range(1, 30, 12), right=False, labels=labels)
    df_2['Total_Charges'] = df_2['Monthly_Bill'] * df_2['Subscription_Length_Months']
    df_2['Senior_Citizen'] = df_2['Age'].apply(lambda age: 1 if age > 60 else 0)
    df_2['Total_Usage_GB'] = pd.to_numeric(df_2['Total_Usage_GB'], errors='coerce')
    df_2['Subscription_Length_Months'] = pd.to_numeric(df_2['Subscription_Length_Months'], errors='coerce')
    df_2['Monthly_Usage_GB'] = df_2['Total_Usage_GB'] / df_2['Subscription_Length_Months']

    
    labels = ["{0} - {1}".format(i, i+19) for i in range(1, df_2['Age'].max(), 20)]
    bins = range(1, df_2['Age'].max() + 20, 20)  # Add 20 to include the upper boundary
    df_2['age_group'] = pd.cut(df_2.Age, bins, right=False, labels=labels)
    df_2.drop(columns=['Subscription_Length_Months', 'Age'], axis=1, inplace=True)
    
    # Create dummy variables for categorical columns
    new_df_dummies = pd.get_dummies(df_2[['Gender', 'Location', 'tenure_group', 'age_group']])
    df_2 = pd.concat([df_2, new_df_dummies],axis='columns')
    df_2.drop(columns=['Gender', 'Location', 'age_group', 'tenure_group'], axis=1, inplace=True) 
    # Make predictions
    single = model.predict(new_df_dummies.tail(1))
    probablity = model.predict_proba(new_df_dummies.tail(1))[:, 1]
    
    # Determine the churn prediction
    if single == 1:
        o1 = "This customer is likely to be churned!!"
        o2 = "Confidence: {}".format(probablity * 100)
    else:
        o1 = "This customer is likely to continue!!"
        o2 = "Confidence: {}".format(probablity * 100)
    
    return render_template('home.html', output1=o1, output2=o2, 
                           query1=request.form['query1'], 
                           query2=request.form['query2'],
                           query3=request.form['query3'],
                           query4=request.form['query4'],
                           query5=request.form['query5'], 
                           query6=request.form['query6'])

if __name__ == "__main__":
    app.debug = True
    app.run()
