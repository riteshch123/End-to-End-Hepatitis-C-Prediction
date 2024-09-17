from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
import pickle
import pandas as pd
import traceback
import joblib  # Use joblib for loading the pipeline

# Load the trained model
with open('best_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load the fitted preprocessing pipeline
pipeline = joblib.load('pipeline.pkl')

app = Flask(__name__)
CORS(app)  # Enable CORS to allow requests from Streamlit app


@app.route('/predict', methods=['POST'])
@cross_origin()
def predict():
    try:
        data = request.get_json(force=True)
        # Convert data into DataFrame
        input_data = pd.DataFrame([data])

        print("Received input data:")
        print(input_data)

        # Ensure 'Category' is not in input_data
        if 'Category' in input_data.columns:
            input_data = input_data.drop(columns=['Category'])
            print("'Category' column was found in input and has been removed.")

        # Preprocess the input data
        X_processed = pipeline.transform(input_data)

        print("Processed input data:")
        print(X_processed)

        # Make prediction
        prediction = model.predict(X_processed)
        # Return the result
        output = {'prediction': int(prediction[0])}
        return jsonify(output)
    except Exception as e:
        # Log the error
        print('Error during prediction:')
        traceback.print_exc()
        # Return error message
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
