import os #import os module to handle file paths
from flask import Flask, request, render_template #import Flask and other necessary modules from flask
import pickle #import pickle module to load the trained model

app = Flask(__name__, template_folder=os.path.join(os.getcwd(), "templates")) #initialize Flask app with template folder

# Load model
with open("model.pkl", "rb") as f: #open the trained model file in read-binary mode
    model = pickle.load(f) #load the model using pickle

flower_names = { 
    0: "Setosa",
    1: "Versicolor",
    2: "Virginica"
} #dictionary to map model output to flower names

# UI Route
@app.route("/") #home page route
def home(): #render home page
     return render_template("index.html") #render index.html template

# UI Prediction Route
@app.route("/predict-ui", methods=["POST"]) #route to handle form submission from UI
def predict_ui(): #function to handle prediction from UI
    features = [
        float(request.form["sepal_length"]),
        float(request.form["sepal_width"]),
        float(request.form["petal_length"]),
        float(request.form["petal_width"])
    ] #extract features from form data and convert to float

    prediction = model.predict([features])[0] #make prediction using the loaded model

    return render_template(
        "index.html",
        prediction=flower_names[prediction]
    ) #render index.html with prediction result

# API (still available)

if __name__ == "__main__": #run the app
    app.run(host="0.0.0.0", port=5000, debug=True) #run the app on all available IPs on port 5000
