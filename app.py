import os
app_root = os.path.dirname(os.path.abspath(__file__))
os.chdir(app_root)


from flask import Flask, render_template, request
import numpy as np
import pickle 

# app = Flask(__name__)
app = Flask(__name__, template_folder="Templets")

# Load the saved model
with open("submission_RF.csv", "rb") as model_file:
    rf_model = pickle.load(model_file)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            # Get input values from the form
            budget = float(request.form['budget'])
            
            # Create an input_data array to use with the model
            input_data = np.array([[budget]])

            # Predict using the model
            rf_pred = rf_model.predict(input_data)

            return render_template('result.html', rf_pred=np.expm1(rf_pred)[0])

        except Exception as e:
            return render_template('error.html', error_message=str(e))

if __name__ == '__main__':
    app.run(debug=True)
