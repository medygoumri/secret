import os
import yfinance as yf
import pandas as pd
import datetime

def download_gold_data(start_date: str, end_date: str, filename: str):
    """
    Downloads historical gold price data from Yahoo Finance
    and saves it to a CSV file.

    Parameters:
      - start_date: Start date in 'YYYY-MM-DD' format.
      - end_date: End date in 'YYYY-MM-DD' format.
      - filename: Path to the CSV file where data will be saved.
    """
    ticker = "GC=F"  # Gold Futures ticker
    print(f"Downloading gold data for {ticker} from {start_date} to {end_date}...")
    data = yf.download(ticker, start=start_date, end=end_date)
    if data.empty:
        print("No data found. Check the ticker or date range.")
    else:
        data.to_csv(filename)
        print(f"Data successfully saved to {filename}")

if __name__ == "__main__":
    # Ensure the data directory exists
    data_folder = "data"
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)
    
    # Define date range (e.g., last 3 years)
    end_date = datetime.datetime.today().strftime("%Y-%m-%d")
    start_date = (datetime.datetime.today() - datetime.timedelta(days=3*365)).strftime("%Y-%m-%d")
    
    output_file = os.path.join(data_folder, "gold_data.csv")
    
    download_gold_data(start_date, end_date, output_file)
