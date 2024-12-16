import requests
import pandas as pd

def fetch_data(api_key):
    url = "https://www.alphavantage.co/query"
    params = {
        "function": "WTI",
        "interval": "daily",
        "datatype": "json",
        "apikey": api_key
    }
    response = requests.get(url, params=params)
    data = response.json()
    print(response.json())
    # Convert to DataFrame and process
    df = pd.DataFrame(data['data']).set_index('date')  # Adjust based on API response
    
    df = df.sort_index()  # Sort by date
    df = df.ffill()  # Fill missing values forward
    return df
#H2KSZCLZNY6TP1MZ Is alphavantage API-key
#H2KSZCLZNY6TP1MZ
API_key = "BH6RFKC1GD7UUAAU"
results = fetch_data(API_key)
results = results.ffill()
#print(results[:5])
results.to_json("oil_prices.json")  # Save as JSON file
