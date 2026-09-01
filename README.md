# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Load the spam mail dataset and preprocess the email text.
2.Convert the email text into numerical features using TF-IDF Vectorization.
3.Train the SVM classifier using the training dataset.
4.Predict and classify emails as Spam or Not Spam (Ham) and evaluate the model accuracy.

## Program:
```
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix

# Load dataset
data = pd.read_csv("spam.csv", encoding="latin-1")

data = data[["v1", "v2"]]
data.columns = ["label", "message"]

# Convert labels
data["label"] = data["label"].map({"ham": 0, "spam": 1})

# Features and target
X = data["message"]
y = data["label"]

# Convert text to numbers
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train SVM
model = SVC(kernel="linear")
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Plot
plt.imshow(cm)
plt.colorbar()

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("SVM Spam Detection")

plt.xticks([0, 1], ["Ham", "Spam"])
plt.yticks([0, 1], ["Ham", "Spam"])

plt.show()

/*
Program to implement the SVM For Spam Mail Detection..
Developed by: Selshiya F
RegisterNumber: 212224060241
*/
```

## Output:

<img width="428" height="308" alt="image" src="https://github.com/user-attachments/assets/b9ebc714-08a5-4e55-9019-9c039542b7f0" />

## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
