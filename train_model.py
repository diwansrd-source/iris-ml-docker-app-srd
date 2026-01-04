from sklearn.datasets import load_iris #load iris dataset 
from sklearn.model_selection import train_test_split #load train_test_split package from sklearn.model selection library
from sklearn.linear_model import LogisticRegression #import LogisticRegression from sklearn.linear model library
import pickle #import pickle library to save the model

# Load dataset
iris = load_iris()
X = iris.data #storing features - sepal length, sepal width, petal length, petal width
y = iris.target #storing target variable - species of iris flower (outcomes)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)  #splitting data into training and testing sets with 20% data for testing

# Train model
model = LogisticRegression(max_iter=200) #initializing Logistic Regression model with max iterations set to 200
model.fit(X_train, y_train) #fitting the model on training data

# Save model
with open("model.pkl", "wb") as file: #opening a file named model.pkl in write-binary mode
    pickle.dump(model, file) #saving the trained model to the file using pickle

print("Model trained and saved as model.pkl") #confirmation message