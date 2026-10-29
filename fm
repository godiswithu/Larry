
# Assignment No. 4: Machine Learning-Based Recommendation System for Breast Cancer Prognosis


# 1. Import the required libraries

import pandas as pd
# Pandas is used to read and handle the dataset.

from sklearn.model_selection import train_test_split
# This function divides the dataset into training and testing data.

from sklearn.ensemble import RandomForestClassifier
# Random Forest is the Machine Learning algorithm used to predict the tumour category.

from sklearn.metrics import accuracy_score
# This function calculates how many predictions are correct.


# 2. Load the dataset

df = pd.read_csv("data.csv")
# Read the Kaggle CSV file and store the dataset in the variable df.

print("First five rows of the dataset:")
# Display a heading before showing the dataset.

print(df.head())
# Display the first five rows to see what the dataset contains.


# 3. Clean the dataset

df = df.drop(columns=["id", "Unnamed: 32"], errors="ignore")
# Remove unnecessary columns. errors="ignore" avoids an error if a column is missing.

df = df.dropna()
# Remove rows that contain missing values.


# 4. Separate input features and output

X = df.drop("diagnosis", axis=1)
# X contains the tumour measurements used as input features.
# axis=1 means we are removing a column.

y = df["diagnosis"].map({"B": 0, "M": 1})
# y contains the correct diagnosis.
# B is converted to 0 (Benign), and M is converted to 1 (Malignant).


# 5. Split the dataset into training and testing data

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    # Use 20% of the data for testing and 80% for training.

    random_state=42,
    # Keep the split the same each time the code runs.

    stratify=y
    # Maintain a similar proportion of both diagnosis categories.
)
# X_train and y_train are used to train the model.
# X_test and y_test are used to evaluate the model.


# 6. Create and train the Machine Learning model

model = RandomForestClassifier(random_state=42)
# Create a Random Forest Classifier.

model.fit(X_train, y_train)
# Train the model using the training measurements and their known diagnoses.


# 7. Test the model and calculate accuracy

predictions = model.predict(X_test)
# Predict the diagnoses of samples in the testing data.

accuracy = accuracy_score(y_test, predictions)
# Compare predicted diagnoses with actual diagnoses to calculate accuracy.

print("Model Accuracy:", round(accuracy * 100, 2), "%")
# Display accuracy as a percentage rounded to two decimal places.


# 8. Predict one sample

sample = X_test.iloc[[0]]
# Select the first sample from the testing data.

result = model.predict(sample)[0]
# Predict its category and store the result as 0 or 1.

print("\nPredicted Category:")
# Display the heading for the prediction.

if result == 0:
    print("Benign")
    # If the result is 0, display Benign.
else:
    print("Malignant")
    # Otherwise, display Malignant.


# 9. Generate a recommendation

print("\nRecommendation:")
# Display the heading for the recommendation.

if result == 0:
    # Check whether the predicted category is Benign.

    print("Discuss the result with a qualified healthcare professional.")
    # Display a general suggestion for professional review.

    print("Follow the professional's advice about appropriate monitoring.")
    # Suggest following professional advice about further monitoring.

else:
    # This block runs if the predicted category is Malignant.

    print("Seek prompt evaluation by a qualified healthcare professional.")
    # Display a general suggestion to seek professional evaluation.

    print("Further assessment is needed to determine the diagnosis and prognosis.")
    # Explain that more assessment is needed to establish the diagnosis and prognosis.


# Note: This is an educational prototype, not a medical diagnostic tool.
# The model predicts benign or malignant; it does not predict survival or recurrence.
# The recommendation is based on simple if-else rules, not a learned recommendation algorithm.




The main aim of my project is to use Machine Learning to classify breast tumour samples as benign or malignant and provide a general recommendation based on the predicted result.
For this project, I used the Breast Cancer Wisconsin dataset from Kaggle. It contains different tumour measurements, such as radius, texture, perimeter, and area.
First, I loaded the dataset using the Pandas library and cleaned it by removing unnecessary columns and missing values. Then, I separated the input features from the target diagnosis.
After that, I divided the dataset into 80% training data and 20% testing data. I used the Random Forest Classifier algorithm to train the model and predict the tumour category.
My model achieved 97.37% accuracy on the test data. Finally, I added a simple recommendation component using if-else conditions. It displays a general next-step suggestion based on the predicted category.


the program loads the Breast Cancer Wisconsin dataset from Kaggle and displays the first five rows. The dataset contains tumour measurements and their actual diagnoses.
Next, I clean the data and divide it into training and testing sets. I train a Random Forest Classifier using 80% of the data and evaluate it on the remaining 20%.
My model achieved 97.37% accuracy on the test data. Then, I selected one sample from the test set, and the model predicted its category as benign.
Finally, the program displays a general recommendation based on the predicted category. This recommendation uses simple if-else conditions. The project is an educational prototype and is not intended for actual medical diagnosis.”
https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data?resource=download
