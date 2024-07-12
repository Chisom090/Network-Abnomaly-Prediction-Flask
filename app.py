# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 22:17:01 2024

@author: chisom
"""

from flask import Flask, render_template, request, jsonify, make_response
import pandas as pd
import pickle
import os

app = Flask(__name__)

# Define paths for model and preprocessors
model_path = 'knn_model.pkl'

# Check if model file exists
if not os.path.exists(model_path):
    raise FileNotFoundError("Model file not found")

# Load model
with open(model_path, 'rb') as f:
    knn_model = pickle.load(f)

# Mapping for human-readable descriptions of attack types
descriptions = {
    "normal": "Normal network activity",
    "neptune": "Neptune DoS attack",
    "ipsweep": "IP address sweep (reconnaissance)",
    "satan": "SATAN scan tool",
    "portsweep": "Port sweep (reconnaissance)",
    "smurf": "Smurf DDoS attack",
    "nmap": "Nmap network scan",
    "teardrop": "Teardrop attack",
    "back": "Back Orifice attack",
    "warezclient": "Warez client activity",
    "guess_passwd": "Password guessing attack",
    "buffer_overflow": "Buffer overflow attack",
    "warezmaster": "Warez master activity",
    "ftp_write": "FTP write attack",
    "multihop": "Multi-hop attack",
    "rootkit": "Rootkit attack",
    "imap": "IMAP attack"
}

# Function to preprocess uploaded file
def preprocess_data(df):
    expected_columns = [
        'protocol_type', 'service', 'flag', 'num_compromised', 'root_shell',
        'su_attempted', 'num_root', 'num_file_creations', 'num_shells',
        'num_access_files', 'num_outbound_cmds', 'is_host_login',
        'is_guest_login', 'count', 'srv_count', 'serror rate',
        'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate',
        'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count',
        'dst_host_srv_count', 'dst_host_same_srv_rate',
        'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
        'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
        'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
        'dst_host_srv_rerror_rate'
    ]

    if list(df.columns) != expected_columns:
        # Log mismatch and apply preprocessing
        print("Column mismatch found, applying preprocessing")
        df = df.reindex(columns=expected_columns, fill_value=0)
    return df

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        if file.filename.endswith('.csv'):
            df = pd.read_csv(file)
        elif file.filename.endswith('.json'):
            df = pd.read_json(file)
        else:
            return jsonify({'error': 'Unsupported file format. Please upload a CSV or JSON file.'}), 400

        # Ensure the DataFrame columns match the expected columns of the model
        df_processed = preprocess_data(df)

        # Return a response indicating prediction process has started
        response = {'message': 'Prediction process started...'}

        # Make predictions
        predictions = knn_model.predict(df_processed)

        # Map predictions to human-readable descriptions
        results = []
        for prediction in predictions:
            result = {
                "class": prediction,
                "description": descriptions.get(prediction, "Unknown attack type")
            }
            results.append(result)

        # Convert results to CSV string
        results_df = pd.DataFrame(results)
        csv_output = results_df.to_csv(index=False)

        # Prepare response with CSV content
        response = make_response(csv_output)
        response.headers['Content-Type'] = 'text/csv'
        response.headers['Content-Disposition'] = 'attachment; filename=predictions.csv'

        return response

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
