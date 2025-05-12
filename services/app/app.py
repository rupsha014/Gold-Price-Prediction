from flask import Flask, jsonify
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from joblib import load
from copy import deepcopy as dc
from flask import Flask
from flask_cors import CORS

app = Flask(__name__)
cors = CORS(app)


# Define the LSTM model class
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_stacked_layers):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_stacked_layers = num_stacked_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_stacked_layers,
                            batch_first=True)

        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_stacked_layers, batch_size, self.hidden_size, dtype=torch.float32).to(x.device)
        c0 = torch.zeros(self.num_stacked_layers, batch_size, self.hidden_size, dtype=torch.float32).to(x.device)

        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

@app.route("/")
def model_prediction():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the model and scaler
    model = LSTM(1, 4, 1)
    model.load_state_dict(torch.load('LSTM-price-prediction-gold-model.pt', map_location=device))
    model.to(device)
    model.eval()

    scaler = load('LSTM-price-prediction-gold-scaler.joblib')

    # Few Parameters of the Model
    lookback = 7
    num_predictions = 7

    # Load the test dataset
    X_test = np.load('X_test.npy')

    # Extract the last known values from the test set
    last_known_prices = X_test[-1, :, :].reshape((1, lookback, -1))

    # Convert to tensor and move to device using clone().detach()
    current_sequence = torch.tensor(last_known_prices, dtype=torch.float32).clone().detach().to(device)

    # Make future predictions
    future_predictions = []

    with torch.no_grad():
        for _ in range(num_predictions):
            # Make a prediction
            prediction = model(current_sequence)

            # Store the prediction
            future_predictions.append(prediction.item())

            # Update the current sequence
            prediction_reshaped = prediction.view(1, 1, 1)  # Reshape to (1, 1, 1)
            current_sequence = torch.cat((current_sequence[:, 1:, :], prediction_reshaped), dim=1)

    # Inverse transform the future predictions
    dummies = np.zeros((num_predictions, lookback + 1))  # Create a dummy array with the same number of features
    dummies[:, 0] = future_predictions  # Place future predictions in the first column
    dummies = scaler.inverse_transform(dummies)  # Inverse transform using the scaler
    future_predictions = dc(dummies[:, 0])  # Extract the inverse transformed predictions

    # Create a DataFrame for visualization
    future_dates = pd.date_range(start='2024-06-13', periods=num_predictions)
    df = pd.DataFrame({
        'Date': future_dates,
        'Predicted': future_predictions
    })
    df.set_index('Date', inplace=True)

    # Convert the DataFrame to JSON for the response
    result_json = df.to_json()

    return jsonify(result_json)

if __name__ == "__main__":
    app.run(debug=True)
