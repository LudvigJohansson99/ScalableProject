from transformers import TimeSeriesTransformerForPrediction
from transformers import TimeSeriesTransformerConfig
from torch.optim import Adam
import torch
import pandas as pd
import json
import os


API_key = "H2KSZCLZNY6TP1MZ"
data = pd.read_json("oil_prices.json")
dates = data.index.tolist()
#print(dates[:5])  # Print the first 5 dates
day = data['day'] = data.index.day
month = data['month'] = data.index.month  # 1 (January) to 12 (December)
year = data['year'] = data.index.year  # E.g., 1986

# Extract past and future time features
context_length = 40
prediction_length = 7
lags_sequence = [1, 2, 3, 4, 5, 6, 7]
data['value'] = pd.to_numeric(data['value'], errors='coerce')
data['value'].fillna(method='ffill', inplace=True)

config = TimeSeriesTransformerConfig(
    prediction_length=prediction_length,
    context_length=context_length,
    input_size=1,
    lags_sequence=lags_sequence,
    num_time_features=3,  # day, month, year
    scaling="mean",
    encoder_layers=6,
    decoder_layers=6,
    encoder_ffn_dim=128,
    decoder_ffn_dim=128,
)

past_values = torch.tensor(
    data['value'].iloc[-(context_length + max(lags_sequence)):].values,
    dtype=torch.float32
).unsqueeze(0)

# Slice the future values for the prediction window
future_values = torch.tensor(
    data['value'].iloc[-prediction_length:].values, 
    dtype=torch.float32
).unsqueeze(0)

# Extract time features for the past and future
past_time_features = torch.tensor(
    data[['day', 'month', 'year']].iloc[-(context_length + max(lags_sequence)):].values,
    dtype=torch.float32
).unsqueeze(0)

future_time_features = torch.tensor(
    data[['day', 'month', 'year']].iloc[-prediction_length:].values, 
    dtype=torch.float32
).unsqueeze(0)

# Observed mask (all observed in this case)
past_observed_mask = torch.ones_like(past_values, dtype=torch.bool)



model = TimeSeriesTransformerForPrediction(config)
model.train()

optimizer = Adam(model.parameters(), lr=1e-4)

epochs = 10000
for epoch in range(epochs):
    optimizer.zero_grad()

    outputs = model(
        past_values=past_values,
        past_time_features=past_time_features,
        future_values=future_values,
        future_time_features=future_time_features,
        past_observed_mask = past_observed_mask
    )

    # Calculate loss and backpropagate
    loss = outputs.loss
    loss.backward()
    optimizer.step()
    #print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}")
total_length = len(data['value'])
start_idx = total_length - (context_length + prediction_length + max(lags_sequence))
end_idx = total_length - prediction_length

past_values_pred = torch.tensor(
    data['value'].iloc[start_idx:end_idx].values, dtype=torch.float32
).unsqueeze(0)

past_time_features_pred = torch.tensor(
    data[['day', 'month', 'year']].iloc[start_idx:end_idx].values, dtype=torch.float32
).unsqueeze(0)

# Generate future time features for the next 7 days
last_date = data.index[-1]
future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=prediction_length)   

future_time_features_pred = torch.tensor(
    [[date.day, date.month, date.year] for date in future_dates], 
    dtype=torch.float32
).unsqueeze(0)

past_observed_mask_pred = torch.ones_like(past_values_pred, dtype=torch.bool)

# Make predictions
predictions = model.generate(
    past_values=past_values_pred,
    past_time_features=past_time_features_pred,
    future_time_features=future_time_features_pred,
    past_observed_mask=past_observed_mask_pred,
)

# Extract the final prediction
predicted_values_next_7 = predictions.sequences[-1][-1]
predicted_values_next_7 = predicted_values_next_7.tolist()
dates_next_7 = [date.strftime('%Y-%m-%d') for date in future_dates]
#print("future_dates_str", dates_next_7)

true_values_last_21 = data['value'].iloc[-21:].values
dates_last_21 = data.index[-21:]

PREDICTIONS_FILE = "all_time_predictions.json"
if os.path.exists(PREDICTIONS_FILE):
    with open(PREDICTIONS_FILE, "r") as f:
        try:
            all_predictions = json.load(f)
        except json.JSONDecodeError:
            all_predictions = {}  # Handle case where file is empty or corrupted
else:
    all_predictions = {}
new_predictions_dict = {date.strftime('%Y-%m-%d'): value for date, value in zip(future_dates, predicted_values_next_7)}

for date, value in new_predictions_dict.items():
    all_predictions[date] = value  # Overwrite existing values or add new ones

# Save updated predictions back to the JSON file
with open(PREDICTIONS_FILE, "w") as f:
    json.dump(all_predictions, f, indent=4)

#print(f"Updated predictions saved to {PREDICTIONS_FILE}")
